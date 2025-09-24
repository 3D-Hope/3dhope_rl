from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F

from omegaconf import DictConfig

from steerable_scene_generation.algorithms.common.dataclasses import SceneVecDescription
from steerable_scene_generation.datasets.scene.scene import SceneDataset

from .scene_diffuser_base_continous import SceneDiffuserBaseContinous


def compute_attribute_weighted_ddpm_loss(
    predicted_noise: torch.Tensor,
    noise: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    log_fn: Callable | None = None,
) -> torch.Tensor:
    """
    Computes the attribute-weighted loss for DDPM training.

    Args:
        predicted_noise: The predicted noise tensor.
        noise: The target noise tensor.
        scene_vec_desc: The scene vector descriptor object.
        cfg: Configuration object containing loss weights.
        log_fn: Optional logging function that accepts a dictionary of metrics and a
            batch_size parameter. Expected signature:
            log_fn(metrics_dict: Dict[str, torch.Tensor], batch_size: int) -> None.

    Returns:
        The computed loss value.
    """
    batch_size = predicted_noise.shape[0]

    # Ensure that each object attribute is weighted equally, regardless of its
    # number of parameters.
    translation_loss = F.mse_loss(
        scene_vec_desc.get_translation_vec(predicted_noise),
        scene_vec_desc.get_translation_vec(noise),
    )
    rotation_loss = F.mse_loss(
        scene_vec_desc.get_rotation_vec(predicted_noise),
        scene_vec_desc.get_rotation_vec(noise),
    )
    model_path_loss = F.mse_loss(
        scene_vec_desc.get_model_path_vec(predicted_noise),
        scene_vec_desc.get_model_path_vec(noise),
    )
    loss = (
        cfg.loss.object_translation_attribute_weight * translation_loss
        + cfg.loss.object_rotation_attribute_weight * rotation_loss
        + cfg.loss.object_model_attribute_weight * model_path_loss
    )
    # Normalize the loss for the scaling not to affect the learning rate.
    loss /= (
        cfg.loss.object_translation_attribute_weight
        + cfg.loss.object_rotation_attribute_weight
        + cfg.loss.object_model_attribute_weight
    )

    if log_fn is not None:
        log_fn(
            {
                "training/translation_loss": translation_loss,
                "training/rotation_loss": rotation_loss,
                "training/model_path_loss": model_path_loss,
            },
            batch_size=batch_size,
        )

    return loss


def compute_ddpm_loss(
    predicted_noise: torch.Tensor,
    noise: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    log_fn: Callable | None = None,
) -> torch.Tensor:
    """
    Computes the loss for DDPM training.

    Args:
        predicted_noise: The predicted noise tensor.
        noise: The target noise tensor.
        scene_vec_desc: The scene vector descriptor object.
        cfg: Configuration object containing loss settings.
        log_fn: Optional logging function that accepts a dictionary of metrics and a
            batch_size parameter.

    Returns:
        The computed loss value.
    """
    if cfg.loss.use_separate_loss_per_object_attribute:
        return compute_attribute_weighted_ddpm_loss(
            predicted_noise=predicted_noise,
            noise=noise,
            scene_vec_desc=scene_vec_desc,
            cfg=cfg,
            log_fn=log_fn,
        )
    else:
        loss = F.mse_loss(predicted_noise, noise)
        return loss


class SceneDiffuserTrainerDDPM(SceneDiffuserBaseContinous):
    """
    Class that provides the DDPM training logic.
    """

    def __init__(self, cfg, dataset: SceneDataset):
        """
        cfg is a DictConfig object defined by
        `configurations/algorithm/scene_diffuser_base_continous.yaml`.
        """
        super().__init__(cfg, dataset=dataset)

    def custom_loss_function(
        self, predicted_noise: torch.Tensor, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Custom loss function for your specific scene representation.

        Args:
            predicted_noise: Model predictions (B, N, D)
            noise: Ground truth noise (B, N, D)

        Returns:
            total_loss: Combined loss
            loss_components: Dictionary of individual loss components
        """
        # Define indices for your representation components
        class_indices = list(
            range(0, 22)
        )  # TODO: Adjust based on actual class OHE size
        pos_indices = list(
            range(len(class_indices), len(class_indices) + 3)
        )  # Next 3 dimensions for position
        size_indices = list(
            range(
                len(class_indices) + len(pos_indices),
                len(class_indices) + len(pos_indices) + 3,
            )
        )  # Next 3 dimensions for size
        rot_indices = list(
            range(
                len(class_indices) + len(pos_indices) + len(size_indices),
                len(class_indices) + len(pos_indices) + len(size_indices) + 2,
            )
        )  # Next 2 dimensions for rotation

        # objfeat_indices = list(
        #     range(
        #         len(class_indices)
        #         + len(pos_indices)
        #         + len(size_indices)
        #         + len(rot_indices),
        #         len(class_indices)
        #         + len(pos_indices)
        #         + len(size_indices)
        #         + len(rot_indices)
        #         + 32,
        #     )
        # )  # All dimensions for object features

        # Extract components from your representation using your custom indices
        pred_pos = predicted_noise[..., pos_indices]
        pred_size = predicted_noise[..., size_indices]
        pred_rot = predicted_noise[..., rot_indices]
        pred_class = predicted_noise[..., class_indices]
        # pred_objfeat = predicted_noise[..., objfeat_indices]

        target_pos = noise[..., pos_indices]
        target_size = noise[..., size_indices]
        target_rot = noise[..., rot_indices]
        target_class = noise[..., class_indices]
        # target_objfeat = noise[..., objfeat_indices]

        # Calculate your custom losses
        pos_loss = F.mse_loss(pred_pos, target_pos)
        size_loss = F.mse_loss(pred_size, target_size)
        rot_loss = F.mse_loss(pred_rot, target_rot)
        class_loss = F.mse_loss(pred_class, target_class)
        # objfeat_loss = F.mse_loss(pred_objfeat, target_objfeat)
        # Weight the losses as needed (you can make these configurable)
        pos_weight = 1.0
        size_weight = 1.0
        rot_weight = 1.0
        class_weight = 1.0
        # objfeat_weight = 1.0

        total_loss = (
            pos_weight * pos_loss
            + size_weight * size_loss
            + rot_weight * rot_loss
            + class_weight * class_loss
            # + objfeat_weight * objfeat_loss
        )

        return total_loss, {
            "pos_loss": pos_loss.item(),
            "size_loss": size_loss.item(),
            "rot_loss": rot_loss.item(),
            "class_loss": class_loss.item(),
            # "objfeat_loss": objfeat_loss.item(),
        }

    def loss_function(
        self, predicted_noise: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss function for the DDPM model.
        """
        # Check if we're using a custom dataset and should use custom loss
        import numpy as np

        if hasattr(self.cfg, "custom") and self.cfg.custom.loss:
            # print(f"[Ashok] using custom loss")
            total_loss, loss_components = self.custom_loss_function(
                predicted_noise, noise
            )

            # Log component losses
            for loss_name, loss_value in loss_components.items():
                self.log(f"train/{loss_name}", loss_value, prog_bar=True)

            return total_loss

        # Original loss calculation
        return compute_ddpm_loss(
            predicted_noise=predicted_noise,
            noise=noise,
            scene_vec_desc=self.scene_vec_desc,
            cfg=self.cfg,
            log_fn=self.log_dict,
        )

    def reset_continuous_or_discrete_part(
        self, new: torch.Tensor, old: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset the continuous or discrete part in `new` with `old`.

        Args:
            new (torch.Tensor): The new vector of shape (B, N, V).
            old (torch.Tensor): The old vector of shape (B, N, V).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of the `new` with the continuous
                or discrete part from `old` and the mask of shape (B, N, V) where ones
                correspond to the part to keep.
        """
        if self.cfg.continuous_discrete_only.discrete_only:
            # Only denoise the discrete part.
            mask = torch.concatenate(
                [
                    torch.zeros(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.ones(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(new.device)
        else:
            # Only denoise the continuous part.
            mask = torch.concatenate(
                [
                    torch.ones(
                        self.scene_vec_desc.translation_vec_len
                        + len(self.scene_vec_desc.rotation_parametrization)
                    ),
                    torch.zeros(self.scene_vec_desc.model_path_vec_len),
                ]
            ).to(new.device)

        # Expand the mask to match the shape of new.
        mask_expanded = (
            mask.unsqueeze(0).unsqueeze(0).expand(new.shape)
        )  # Shape (B, N, V)

        # Reset to old where the mask is zero.
        new_reset = new * mask_expanded + old * (1 - mask_expanded)
        return new_reset, mask_expanded

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        phase: str = "training",
        use_ema: bool = False,
    ) -> torch.Tensor:
        """
        DDPM forward pass. Normal diffusion training with maximum likelihood objective.
        Returns the loss.
        """
        scenes = batch["scenes"]

        # Sample noise to add to the scenes.
        noise = torch.randn(scenes.shape).to(self.device)  # Shape (B, N, V)

        # Sample a timestep for each scene.
        timesteps = (
            torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (scenes.shape[0],),
            )
            .long()
            .to(self.device)
        )

        # Add noise to the scenes.
        noisy_scenes = self.noise_scheduler.add_noise(
            scenes, noise, timesteps
        )  # Shape (B, N, V)

        if (
            self.cfg.continuous_discrete_only.continuous_only
            or self.cfg.continuous_discrete_only.discrete_only
        ):
            # Don't add noise to the continuous or discrete part.
            noisy_scenes, mask = self.reset_continuous_or_discrete_part(
                new=noisy_scenes, old=scenes
            )

        predicted_noise = self.predict_noise(
            noisy_scenes=noisy_scenes,
            timesteps=timesteps,
            cond_dict=batch,
            use_ema=use_ema,
        )  # Shape (B, N, V)

        if (
            self.cfg.continuous_discrete_only.continuous_only
            or self.cfg.continuous_discrete_only.discrete_only
        ):
            predicted_noise *= mask
            noise *= mask

        # Compute loss.
        # Check if we're using a custom dataset and should use custom loss
        if (
            hasattr(self.cfg, "dataset_name")
            and self.cfg.dataset_name == "custom_scene"
        ):
            # Custom loss calculation for your representation
            loss, loss_components = self.custom_loss_function(predicted_noise, noise)

            # Log component losses
            for loss_name, loss_value in loss_components.items():
                self.log(f"train/{loss_name}", loss_value, prog_bar=True)
        else:
            # Original loss calculation
            loss = self.loss_function(predicted_noise, noise)

        return loss
