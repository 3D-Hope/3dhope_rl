from typing import Dict

import torch

from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .trainer_rl import SceneDiffuserTrainerRL


class SceneDiffuserTrainerScore(SceneDiffuserTrainerRL):
    """
    Class that provides REINFORCE (score function gradient estimator) training logic.
    This corresponds to DPPO_{SF} (https://arxiv.org/abs/2305.13301).
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)
        self.incremental_training = self.cfg.ddpo.incremental_training
        self.joint_training = self.cfg.ddpo.joint_training
        if self.incremental_training and self.joint_training:
            raise ValueError(
                "Cannot have both incremental_training and joint_training set to True."
            )
        if self.incremental_training:
            self.training_steps = self.cfg.ddpo.training_steps_start # TODO: make this configurable
            self.incremental_n_timesteps_to_sample = [10, 25, 40, 65, 80, 95, 110, 125, 150]
            self.training_steps_per_increment = 1500

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        DDPO-like (https://arxiv.org/abs/2305.13301) forward pass. Training with RL.
        Returns the loss.
        """

        # Get diffusion trajectories.
        (
            trajectories,  # Shape (B, T+1, N, V)
            trajectories_log_props,  # Shape (B, T)
            cond_dict,
        ) = self.generate_trajs_for_ddpo(
            last_n_timesteps_only=self.cfg.ddpo.last_n_timesteps_only,
            n_timesteps_to_sample=self.incremental_n_timesteps_to_sample[self.training_steps // self.training_steps_per_increment] if self.incremental_training else self.cfg.ddpo.n_timesteps_to_sample,
            batch=batch,
            incremental_training=self.incremental_training,
            joint_training=self.joint_training,
        )
        if self.incremental_training:
            self.training_steps += 1
            if self.training_steps % self.training_steps_per_increment == 0:
                print(f"[Ashok] Incremented training to {self.incremental_n_timesteps_to_sample[self.training_steps // self.training_steps_per_increment]} timesteps.")
        # Remove initial noisy scene.
        trajectories = trajectories[
            :, 1:
        ]  # Shape (B, T, N, V) T=timesteps per sample eg, 150

        # Compute rewards.
        rewards = self.compute_rewards_from_trajs(
            trajectories=trajectories, cond_dict=cond_dict
        )  # Shape (B,)

        # Compute advantages.
        advantages = self.compute_advantages(rewards, phase=phase)  # Shape (B,)

        # REINFORCE loss.
        loss = -torch.mean(torch.sum(trajectories_log_props, dim=1) * advantages)
        print(f"[Ashok] reinforce loss values: {loss.item()}")
        # DDPM loss for regularization.
        if self.cfg.ddpo.ddpm_reg_weight > 0.0:
            # ddpm_loss = self.compute_ddpm_loss(batch)
            ddpm_loss = self.compute_ddpm_loss(cond_dict)
            loss += ddpm_loss * self.cfg.ddpo.ddpm_reg_weight
            print(
                f"[Ashok] reg ddpm loss values: {ddpm_loss.item()*self.cfg.ddpo.ddpm_reg_weight}"
            )

        return loss
