"""
Helpers for the DDPO algorithm (https://arxiv.org/abs/2305.13301).
"""

import math
import multiprocessing

from functools import partial
from typing import List, Optional, Tuple, Union

import torch

from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
from omegaconf import DictConfig
from pydrake.all import QueryObject, SignedDistancePair

from steerable_scene_generation.algorithms.common.dataclasses import (
    PlantSceneGraphCache,
    SceneVecDescription,
)
from steerable_scene_generation.utils.drake_utils import (
    create_plant_and_scene_graph_from_scene_with_cache,
)
from steerable_scene_generation.utils.prompt_following_metrics import (
    compute_prompt_following_metrics,
)

from .inpainting_helpers import (
    generate_empty_object_inpainting_masks,
    generate_physical_feasibility_inpainting_masks,
)


def ddpm_step_with_logprob(
    scheduler: DDPMScheduler,
    model_output: torch.Tensor,
    timestep: int,
    sample: torch.Tensor,
    generator=None,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied and adapted from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py
    to return the log probability of the previous sample. If the previous sample is not
    provided, it is computed from the model output as in the original implementation.
    The style matches the original implementation to facilitate comparison.

    Predict the sample from the previous timestep by reversing the SDE. This function
    propagates the diffusion process from the learned model outputs (most often the
    predicted noise).

    Args:
        scheduler (`diffusers.DDPMScheduler`):
            The scheduler object that contains the parameters of the diffusion process.
        model_output (`torch.Tensor`):
            The direct output from learned diffusion model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.Tensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
        prev_sample (`torch.Tensor`, *optional*): The previous sample. If not provided,
            it is computed from the model output.

    Returns:
        A tuple containing the (predicted) previous sample and the log probability of
        the previous sample.
    """
    assert isinstance(
        scheduler, DDPMScheduler
    ), "scheduler must be an instance of DDPMScheduler"

    t = timestep

    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in [
        "learned",
        "learned_range",
    ]:
        model_output, predicted_variance = torch.split(
            model_output, sample.shape[1], dim=1
        )
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one "
            "of `epsilon`, `sample` or `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (
        alpha_prod_t_prev ** (0.5) * current_beta_t
    ) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = (
        pred_original_sample_coeff * pred_original_sample
        + current_sample_coeff * sample
    )

    # 6. Add noise
    variance_noise = randn_tensor(
        model_output.shape,
        generator=generator,
        device=model_output.device,
        dtype=model_output.dtype,
    )
    if scheduler.variance_type == "fixed_small_log":
        variance = scheduler._get_variance(t, predicted_variance=predicted_variance)
    elif scheduler.variance_type == "learned_range":
        variance = scheduler._get_variance(t, predicted_variance=predicted_variance)
        variance = torch.exp(0.5 * variance)
    else:
        variance = (
            scheduler._get_variance(t, predicted_variance=predicted_variance) ** 0.5
        )

    if prev_sample is None:
        # Don't add noise at t=0.
        prev_sample = (
            pred_prev_sample + variance * variance_noise if t > 0 else pred_prev_sample
        )

    # Log probability of prev_sample (Gaussian distribution).
    log_prob = (
        -((prev_sample.detach() - pred_prev_sample) ** 2) / (2 * (variance**2))
        - torch.log(variance)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # Compute mean log probability over all but batch dimension. This is the combined
    # log probability as the individual elements of xt are independent.
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob


def ddim_step_with_logprob(
    scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: int,
    sample: torch.FloatTensor,
    eta: float = 1.0,
    use_clipped_model_output: bool = False,
    generator=None,
    prev_sample: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied and adapted from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
    to return the log probability of the previous sample. If the previous sample is not
    provided, it is computed from the model output as in the original implementation.
    The style matches the original implementation to facilitate comparison.

    Predict the sample at the previous timestep by reversing the SDE. Core function to
    propagate the diffusion process from the learned model outputs (most often the
    predicted noise).

    Args:
        scheduler (`diffusers.DDIMScheduler`): scheduler object that contains the
            parameters of the diffusion process.
        model_output (`torch.FloatTensor`): direct output from learned diffusion model.
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected" `model_output`
            from the clipped predicted original sample. Necessary because predicted
            original sample is clipped to [-1, 1] when `self.config.clip_sample` is
            `True`. If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output` will
            have not effect.
        generator: random number generator.
        prev_sample (`torch.Tensor`, *optional*): The previous sample. If not provided,
            it is computed from the model output.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the (predicted) previous
            sample and the log probability of the previous sample.
    """
    assert isinstance(
        scheduler, DDIMScheduler
    ), "scheduler must be an instance of DDIMScheduler"
    assert eta >= 0.0, "eta must be non-negative"
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' "
            "after creating the scheduler"
        )

    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else scheduler.final_alpha_cumprod
    )

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (
            beta_prod_t**0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (
            beta_prod_t**0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of "
            "`epsilon`, `sample`, or `v_prediction`"
        )

    # 4. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    variance = scheduler._get_variance(timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 6. compute "direction pointing to x_t" of formula (12) from
    # https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 7. compute x_t without "random noise" of formula (12) from
    # https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either "
            "`generator` or `prev_sample` stays `None`."
        )

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # Log probability of prev_sample (Gaussian distribution).
    std_dev_t = torch.clip(std_dev_t, min=1e-6)
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # Compute mean log probability over all but batch dimension. This is the combined
    # log probability as the individual elements of xt are independent.
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample, log_prob


def compute_non_penetration_reward(
    scene: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cache: Optional[PlantSceneGraphCache] = None,
    return_updated_cache: bool = True,
) -> Union[float, Tuple[float, PlantSceneGraphCache]]:
    """
    Get the non-penetration reward for a scene. The reward is the sum of the negative
    distances between the objects in the scene. If the scene is collision-free, the
    reward is 0.0 (the best possible reward).

    Args:
        scene (torch.Tensor): The unormalized scene to score. The scene is represented as
            a tensor of shape (num_objects, num_features).
        scene_vec_desc (SceneVecStructureDescription): The description of the scene
            vector structure.
        cache (Optional[PlantSceneGraphCache]): The PlantSceneGraphCache. If None or if
            the objects in the scene have changed, the plant and scene graph are
            recreated.
        return_updated_cache (bool): If True, the updated PlantSceneGraphCache is
            returned.

    Returns:
        The non-penetration reward for the scene. If return_updated_cache is True, the
        updated PlantSceneGraphCache is also returned.
    """
    # Create the diagram for the scene.
    cache, context, _ = create_plant_and_scene_graph_from_scene_with_cache(
        scene=scene, scene_vec_desc=scene_vec_desc, cache=cache
    )
    scene_graph = cache.scene_graph
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)
    query_object: QueryObject = scene_graph.get_query_output_port().Eval(
        scene_graph_context
    )

    # Get all negative distances between the objects in the scene. These are the
    # penetration distances.
    signed_distance_pairs: List[
        SignedDistancePair
    ] = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.0)
    distances = [pair.distance for pair in signed_distance_pairs]

    # Compute the non-penetration reward.
    reward = sum(distances) if len(distances) > 0 else 0.0

    if return_updated_cache:
        return reward, cache
    return reward


def non_penetration_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    num_workers: int = 1,
    cache: Optional[PlantSceneGraphCache] = None,
    return_updated_cache: bool = False,
) -> torch.Tensor:
    """
    Compute the non-penetration reward for a scene. The reward is the sum of the
    negative distances between the objects in the scene. If the scene is collision-free,
    the reward is 0.0 (the best possible reward).

    Args:
        scenes (torch.Tensor): The unormalized scenes to score. The scenes are
            represented as a tensor of shape (batch_size, num_objects, num_features).
        scene_vec_desc (SceneVecStructureDescription): The description of the scene
            vector structure.
        num_workers (int): The number of workers to use for parallel processing. Note
            that using multiple workers prevents the use of the cache and thus might be
            slower if all the scenes contain the same objects.
        cache (Optional[PlantSceneGraphCache]): The PlantSceneGraphCache. If None or if
            the objects in the scene have changed, the plant and scene graph are
            recreated.
        return_updated_cache (bool): If True, the updated PlantSceneGraphCache is
            returned.

    Returns:
        The non-penetration reward for the scenes. If return_updated_cache is True, the
        updated PlantSceneGraphCache is also returned.
    """
    device = scenes.device
    scenes = scenes.cpu().detach().numpy()

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            rewards = pool.map(
                partial(
                    compute_non_penetration_reward,
                    scene_vec_desc=scene_vec_desc,
                    cache=cache,
                    return_updated_cache=False,
                ),
                scenes,
            )
            rewards = torch.tensor(rewards, device=device)
    else:
        rewards = torch.zeros(scenes.shape[0], device=device)
        for i, scene in enumerate(scenes):
            rewards[i], cache = compute_non_penetration_reward(
                scene=scene, scene_vec_desc=scene_vec_desc, cache=cache
            )

    if return_updated_cache:
        return rewards, cache
    print("[Ashok] non-penetration rewards:", rewards)
    return rewards


def object_number_reward(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg=None
) -> torch.Tensor:
    """
    Compute the object number reward for a scene. The reward is the number of objects
    in the scene.

    Args:
        scenes (torch.Tensor): The unormalized scenes to score of shape (B, N, V).
        scene_vec_desc (SceneVecStructureDescription): The description of the scene
            vector structure.

    Returns:
        The object number reward for the scenes of shape (B,).
    """
    rewards = torch.zeros(scenes.shape[0], device=scenes.device)  # Shape (B,)
    for i, scene in enumerate(scenes):
        # Count non-empty objects.
        if cfg is None or not cfg.custom.use:
            num_objects = sum(
                scene_vec_desc.get_model_path(obj) is not None for obj in scene
            )
        else:
            # Custom format with 30 dimensions and first 22 are class labels
            num_objects = (
                (scene[:, : cfg.custom.num_classes].argmax(dim=-1) != 21).sum().item()
            )
        rewards[i] = num_objects
    print("[Ashok] object no rewards:", rewards)
    return rewards


def iou_reward(scenes: torch.Tensor, scene_diffuser, cfg) -> torch.Tensor:
    """
    Compute the IoU reward for scenes. The reward is the negative of the average IoU
    between valid objects in each scene. This encourages scenes with less object overlap.

    Args:
        scenes (torch.Tensor): The unormalized scenes to score of shape (B, N, V).
        scene_diffuser: The scene diffuser model with IoU calculation function.
        cfg: Optional configuration object.

    Returns:
        The IoU reward for the scenes of shape (B,).
    """
    if scene_diffuser is None:
        raise ValueError("scene_diffuser must be provided for IoU reward calculation")

    # Convert list of scenes to batch tensor if needed
    if isinstance(scenes, list):
        scene_batch = torch.stack(scenes, dim=0)
    else:
        scene_batch = scenes

    iou_values = scene_diffuser.bbox_iou_regularizer(
        recon=scene_batch, num_classes=cfg.custom.num_classes, using_as_reward=True
    )
    # TODO: AVOID SELF IOU
    # Convert to list for compatibility if needed
    if isinstance(scenes, list):
        return iou_values.detach().cpu().tolist()
    # print("[Ashok] IoU values:", iou_values.shape)
    rewards = iou_values  # unnormalized raw iou values
    print("[Ashok] IoU rewards:", rewards)
    return rewards


def two_beds_reward(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg=None
) -> torch.Tensor:
    """
    Reward = 1 if there are exactly 2 beds in the scene, else 0.
    """
    rewards = torch.zeros(scenes.shape[0], device=scenes.device)  # Shape (B,)
    for i, scene in enumerate(scenes):
        # Sum probabilities for bed objects
        beds_idx = [8, 15, 11]
        print(
            "[Ashok] scene[:, : cfg.custom.num_classes]:",
            scene[:, : cfg.custom.num_classes],
        )
        # Sum the probabilities of bed classes across all objects
        bed_probabilities = scene[
            :, beds_idx
        ].sum()  # TODO: need better reward. this naive approaach will simply lead to higher probs for bed classes not 100% but generally higher.
        rewards[i] = bed_probabilities.item()
    print("[Ashok] 2 beds rewards:", rewards)
    return rewards


def has_sofa_reward(
    scenes: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg=None
) -> torch.Tensor:
    """
    Reward = 1 if there is a sofa in the scene, else 0.
    """
    rewards = torch.zeros(scenes.shape[0], device=scenes.device)  # Shape (B,)
    # print("[Ashok] scenes: ", scenes)
    for i, scene in enumerate(scenes):
        # print(f"class probs in reward {scene[:, : cfg.custom.num_classes]}")
        # Check if sofa class is present
        sofa_idx = 17
        has_sofa = (scene[:, sofa_idx] > 0).any().item()
        rewards[i] = float(has_sofa)
    print("[Ashok] has sofa rewards:", rewards)
    return rewards


def composite_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg=None,
    room_type: str = "bedroom",
    importance_weights: dict = None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute composite reward using multiple physics-based constraints.

    This function uses the get_composite_reward from physical_constraint_rewards.commons
    which handles:
    - Gravity following (objects should rest on ground)
    - Non-penetration (no overlapping objects)
    - Must-have furniture (room-specific requirements)
    - Object count (realistic scene density)

    All rewards are normalized to [-1, 0] range, then weighted by importance.

    Args:
        scenes (torch.Tensor): The scenes to score of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene vector structure.
        cfg (DictConfig, optional): Configuration object.
        room_type (str): Type of room for must-have furniture ('bedroom', 'living_room', etc.)
        importance_weights (dict, optional): Importance multipliers for each reward component.
            If None, uses defaults from config or commons.py.

    Returns:
        tuple: (total_rewards, reward_components)
            - total_rewards: Tensor of shape (B,) with combined rewards
            - reward_components: Dict with individual reward values for logging
    """
    from physical_constraint_rewards.commons import get_composite_reward

    # Get number of classes from config
    num_classes = cfg.custom.num_classes if cfg and hasattr(cfg, "custom") else 22

    # Use importance weights from config if not provided
    if importance_weights is None and cfg and hasattr(cfg, "ddpo"):
        if hasattr(cfg.ddpo, "composite_reward"):
            importance_weights = dict(cfg.ddpo.composite_reward.importance_weights)

    # Compute composite reward
    total_rewards, reward_components = get_composite_reward(
        scenes=scenes,
        num_classes=num_classes,
        importance_weights=importance_weights,
        room_type=room_type,
    )

    print(f"[Ashok] Composite reward components:")
    for name, values in reward_components.items():
        print(
            f"  {name}: mean={values.mean().item():.4f}, std={values.std().item():.4f}"
        )
    print(
        f"[Ashok] Total composite rewards: mean={total_rewards.mean().item():.4f}, std={total_rewards.std().item():.4f}"
    )

    return total_rewards, reward_components


def composite_plus_task_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg=None,
) -> tuple[torch.Tensor, dict]:
    """
    Compute composite reward (general scene quality) plus task-specific reward.
    
    This combines:
    - Composite reward: gravity + non-penetration + must-have + object count
    - Task-specific reward: has_sofa, two_beds, etc.
    
    Final reward = composite_reward + task_weight * task_reward
    
    Args:
        scenes (torch.Tensor): The scenes to score of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene vector structure.
        cfg (DictConfig): Configuration object (required for task settings).
    
    Returns:
        tuple: (total_rewards, reward_components)
            - total_rewards: Tensor of shape (B,) with combined rewards
            - reward_components: Dict with individual reward values for logging
    """
    from physical_constraint_rewards.commons import get_composite_reward
    
    if cfg is None or not hasattr(cfg.ddpo, 'composite_plus_task'):
        raise ValueError("cfg.ddpo.composite_plus_task configuration is required")
    
    task_cfg = cfg.ddpo.composite_plus_task
    
    # Get task-specific settings
    task_reward_type = task_cfg.get('task_reward_type', 'has_sofa')
    task_weight = task_cfg.get('task_weight', 2.0)
    room_type = task_cfg.get('room_type', 'living_room')
    importance_weights = task_cfg.get('importance_weights', None)
    
    # Convert importance_weights from DictConfig to dict if needed
    if importance_weights is not None:
        importance_weights = dict(importance_weights)
    
    # Get number of classes from config
    num_classes = cfg.custom.num_classes if cfg and hasattr(cfg, "custom") else 22
    
    # 1. Compute composite reward (general scene quality)
    composite_total, composite_components = get_composite_reward(
        scenes=scenes,
        num_classes=num_classes,
        importance_weights=importance_weights,
        room_type=room_type,
    )
    
    # 2. Compute task-specific reward
    if task_reward_type == 'has_sofa':
        task_reward = has_sofa_reward(scenes, scene_vec_desc, cfg)
    elif task_reward_type == 'two_beds':
        task_reward = two_beds_reward(scenes, scene_vec_desc, cfg)
    elif task_reward_type == 'has_table':
        # Add table checking (index 18)
        task_reward = torch.zeros(scenes.shape[0], device=scenes.device)
        for i, scene in enumerate(scenes):
            has_table = (scene[:, 18] > 0).any().item()
            task_reward[i] = float(has_table)
    else:
        raise ValueError(f"Unknown task_reward_type: {task_reward_type}")
    
    # 3. Combine rewards
    total_rewards = composite_total + task_weight * task_reward
    
    # 4. Create comprehensive components dict for logging
    reward_components = composite_components.copy()
    reward_components['task_reward'] = task_reward
    reward_components['composite_reward'] = composite_total
    reward_components['total_reward'] = total_rewards
    
    # Print summary
    print(f"[Ashok] Composite + Task Reward:")
    print(f"  Task type: {task_reward_type}, Weight: {task_weight}")
    print(f"  Composite: mean={composite_total.mean().item():.4f}, std={composite_total.std().item():.4f}")
    print(f"  Task: mean={task_reward.mean().item():.4f}, std={task_reward.std().item():.4f}")
    print(f"  Total: mean={total_rewards.mean().item():.4f}, std={total_rewards.std().item():.4f}")
    
    return total_rewards, reward_components


def descale_to_origin(x, minimum, maximum):
    """
    Descale normalized coordinates back to original range.

    Args:
        x: Tensor of shape BxNx3 with values in range [-1, 1]
        minimum: Tensor of shape 3 with minimum values for each dimension
        maximum: Tensor of shape 3 with maximum values for each dimension

    Returns:
        Tensor of shape BxNx3 with values descaled to original range
    """
    x = (x + 1) / 2
    x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
    return x


def mutable_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    reward_func,
    cfg=None,
    class_to_name_map=None,
) -> torch.Tensor:
    """
    Descales generated scenes back to world coordinates and calls a dynamic reward function
    passed as an argument. This allows for on-the-fly reward function creation and usage.

    Args:
        scenes (torch.Tensor): The normalized scenes to score of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene vector structure.
        reward_func (callable): A reward function that takes descaled scenes, class labels,
                               class mapping, and other parameters.
        cfg (DictConfig, optional): Optional configuration object.
        class_to_name_map (dict, optional): Mapping from class indices to object names.
                                          If None, will use default if available in cfg.

    Returns:
        torch.Tensor: The rewards for the scenes of shape (B,).
    """
    # Default class-to-name mapping if not provided
    if class_to_name_map is None:
        if cfg and hasattr(cfg, "class_mapping"):
            class_to_name_map = cfg.class_mapping
        else:
            # Default mapping from the provided data
            class_to_name_map = {
                0: "armchair",
                1: "bookshelf",
                2: "cabinet",
                3: "ceiling_lamp",
                4: "chair",
                5: "children_cabinet",
                6: "coffee_table",
                7: "desk",
                8: "double_bed",
                9: "dressing_chair",
                10: "dressing_table",
                11: "kids_bed",
                12: "nightstand",
                13: "pendant_lamp",
                14: "shelf",
                15: "single_bed",
                16: "sofa",
                17: "stool",
                18: "table",
                19: "tv_stand",
                20: "wardrobe",
            }

    device = scenes.device

    # Extract components from scenes
    num_classes = cfg.custom.num_classes if cfg and hasattr(cfg, "custom") else 22

    # Extract class labels, positions, and sizes
    class_labels = scenes[:, :, :num_classes]  # Shape: B x N x num_classes
    positions = scenes[:, :, num_classes : num_classes + 3]  # Shape: B x N x 3
    sizes = scenes[:, :, num_classes + 3 : num_classes + 6]  # Shape: B x N x 3

    # Descale positions and sizes to original coordinate space
    if cfg and hasattr(cfg, "descale"):
        pos_min = torch.tensor(cfg.descale.position_min, device=device)
        pos_max = torch.tensor(cfg.descale.position_max, device=device)
        size_min = torch.tensor(cfg.descale.size_min, device=device)
        size_max = torch.tensor(cfg.descale.size_max, device=device)
    else:
        # Default min/max values from the provided code
        pos_min = torch.tensor([-2.7625005, 0.045, -2.75275], device=device)
        pos_max = torch.tensor([2.7784417, 3.6248395, 2.8185427], device=device)
        size_min = torch.tensor([0.03998289, 0.02000002, 0.012772], device=device)
        size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)

    descaled_positions = descale_to_origin(positions, pos_min, pos_max)
    descaled_sizes = descale_to_origin(sizes, size_min, size_max)

    # Create a data structure with all necessary information for the reward function
    scene_data = {
        "original_scenes": scenes,
        "class_labels": class_labels,
        "positions": descaled_positions,
        "sizes": descaled_sizes,
        "class_to_name_map": class_to_name_map,
        "num_classes": num_classes,
    }

    # Call the provided reward function
    rewards = reward_func(scene_data, scene_vec_desc, cfg)

    print("[Ashok] dynamic rewards:", rewards)
    return rewards


def prompt_following_reward(
    scenes: torch.Tensor, prompts: list[str], scene_vec_desc: SceneVecDescription
) -> torch.Tensor:
    """
    Compute the prompt following reward for a set of scenes based on their prompts.

    This function calculates the fraction of prompts that are followed correctly
    by the corresponding scenes. It utilizes the `compute_prompt_following_metrics`
    function to derive the necessary metrics.

    Args:
        scenes (torch.Tensor): The unormalized scenes to evaluate, of shape (B, N, V).
        prompts (list[str]): A list of textual prompts describing the scenes.
        scene_vec_desc (SceneVecDescription): The scene vector description.

    Returns:
        torch.Tensor: The prompt following rewards of shape (B,), representing the
        fraction of correctly followed prompts.
    """
    if not len(scenes) == len(prompts):
        raise ValueError(
            "The number of scenes and prompts must be the same. "
            f"Got {len(scenes)} scenes and {len(prompts)} prompts."
        )

    prompt_following_metrics = compute_prompt_following_metrics(
        scene_vec_desc=scene_vec_desc, scenes=scenes, prompts=prompts, disable_tqdm=True
    )

    rewards = torch.tensor(
        prompt_following_metrics["per_prompt_following_fractions"],
        device=scenes.device,
    )  # Shape (B,)
    return rewards


def compute_physically_feasible_objects_reward(
    scene: torch.Tensor, scene_vec_desc: SceneVecDescription, cfg: DictConfig
) -> float:
    """
    Compute the number of physically feasible objects reward for a single scene.

    Args:
        scene (torch.Tensor): The unormalized scene to evaluate, of shape (N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene
            vector structure.
        cfg (DictConfig): The configuration for the physical feasibility.

    Returns:
        float: The number of physically feasible objects reward for the scene.
    """
    # Add batch dimension for the single scene
    scene_batch = scene.unsqueeze(0)  # Shape (1, N, V)

    physical_mask, _, _ = generate_physical_feasibility_inpainting_masks(
        scenes=scene_batch,
        scene_vec_desc=scene_vec_desc,
        non_penetration_threshold=cfg.non_penetration_threshold,
        use_sim=cfg.use_sim,
        sim_duration=cfg.sim_duration,
        sim_time_step=cfg.sim_time_step,
        sim_translation_threshold=cfg.sim_translation_threshold,
        sim_rotation_threshold=cfg.sim_rotation_threshold,
        static_equilibrium_distance_threshold=cfg.static_equilibrium_distance_threshold,
    )  # Shape (1, N, V)

    empty_mask, _ = generate_empty_object_inpainting_masks(
        scenes=scene_batch, scene_vec_desc=scene_vec_desc
    )  # Shape (1, N, V)

    combined_mask = torch.logical_or(physical_mask, empty_mask)  # Shape (1, N, V)

    # Invert so that the mask represents the physically feasible objects
    combined_mask_inverted = torch.logical_not(combined_mask)  # Shape (1, N, V)

    # Convert to object-level for reward value consistency
    object_level_masks = combined_mask_inverted.any(dim=2)  # Shape (1, N)

    # The reward is the number of objects that are physically feasible
    reward = object_level_masks.sum().item()  # Scalar

    return reward


def number_of_physically_feasible_objects_reward(
    scenes: torch.Tensor,
    scene_vec_desc: SceneVecDescription,
    cfg: DictConfig,
    num_workers: int = 1,
) -> torch.Tensor:
    """
    Compute the number of physically feasible objects reward for a scene. The reward
    is the number of objects that are physically feasible (non-penetration and
    static equilibrium).

    Args:
        scenes (torch.Tensor): The unormalized scenes to evaluate, of shape (B, N, V).
        scene_vec_desc (SceneVecDescription): The description of the scene
            vector structure.
        cfg (DictConfig): The configuration for the physical feasibility.
        num_workers (int): The number of workers to use for parallel processing.

    Returns:
        The number of physically feasible objects reward for the scenes of shape (B,).
    """
    device = scenes.device
    scenes_cpu = scenes.cpu().detach()

    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            rewards = pool.map(
                partial(
                    compute_physically_feasible_objects_reward,
                    scene_vec_desc=scene_vec_desc,
                    cfg=cfg,
                ),
                scenes_cpu,
            )
            rewards = torch.tensor(rewards, device=device)
    else:
        rewards = torch.zeros(scenes.shape[0], device=device)
        for i, scene in enumerate(scenes_cpu):
            rewards[i] = compute_physically_feasible_objects_reward(
                scene=scene,
                scene_vec_desc=scene_vec_desc,
                cfg=cfg,
            )

    return rewards.float()
