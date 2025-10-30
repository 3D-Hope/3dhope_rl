import torch

idx_to_labels = {
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

ceiling_objects = ["ceiling_lamp", "pendant_lamp"]


def descale_to_origin(x, minimum, maximum):
    """
    x shape : BxNx3
    minimum, maximum shape: 3
    """
    x = (x + 1) / 2
    x = x * (maximum - minimum)[None, None, :] + minimum[None, None, :]
    return x


def descale_pos(positions, pos_min=None, pos_max=None, device="cuda"):
    """
    Descale positions to original coordinates.

    Args:
        positions: Tensor of shape BxNx3
        pos_min: Minimum position values (optional)
        pos_max: Maximum position values (optional)
        device: Device for tensors

    Returns:
        Descaled positions
    """
    if pos_min is None:
        pos_min = torch.tensor([-2.7625005, 0.045, -2.75275], device=device)
    if pos_max is None:
        pos_max = torch.tensor([2.7784417, 3.6248395, 2.8185427], device=device)

    return descale_to_origin(positions, pos_min, pos_max)


def descale_size(sizes, size_min=None, size_max=None, device="cuda"):
    """
    Descale sizes to original dimensions.

    IMPORTANT: The returned sizes are HALF-EXTENTS (sx/2, sy/2, sz/2), not full dimensions.
    This means:
    - For a box centered at (x, y, z) with returned size (sx, sy, sz):
      - The box extends from (x-sx, y-sy, z-sz) to (x+sx, y+sy, z+sz)
      - Full dimensions would be (2*sx, 2*sy, 2*sz)
    - When computing bounding boxes: use size directly, DO NOT divide by 2 again
    - When computing object bottom: y_min = y_center - y_size (not y_center - y_size/2)

    Args:
        sizes: Tensor of shape BxNx3 (normalized)
        size_min: Minimum size values (optional)
        size_max: Maximum size values (optional)
        device: Device for tensors

    Returns:
        Descaled sizes (HALF-EXTENTS)
    """
    if size_min is None:
        size_min = torch.tensor([0.03998289, 0.02000002, 0.012772], device=device)
    if size_max is None:
        size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)

    return descale_to_origin(sizes, size_min, size_max)


def parse_and_descale_scenes(scenes, num_classes=22, parse_only=False):
    """
    Parse scene tensor and descale positions/sizes to world coordinates.

    IMPORTANT: Sizes are HALF-EXTENTS (sx/2, sy/2, sz/2), not full dimensions!
    - For bounding boxes: min = center - size, max = center + size
    - DO NOT divide sizes by 2 again in reward calculations

    Args:
        scenes: Tensor of shape (B, N, 30)
        num_classes: Number of object classes (default: 22)

    Returns:
        dict with keys:
            - one_hot: (B, N, num_classes)
            - positions: (B, N, 3) - world coordinates
            - sizes: (B, N, 3) - world coordinates (HALF-EXTENTS!)
            - orientations: (B, N, 2) - [cos_theta, sin_theta]
            - object_indices: (B, N) - argmax of one_hot
            - is_empty: (B, N) - boolean mask for empty slots
            - device: device of input tensor
    """
    device = scenes.device

    # Parse scene representation
    one_hot = scenes[:, :, :num_classes]
    positions_normalized = scenes[:, :, num_classes : num_classes + 3]
    sizes_normalized = scenes[:, :, num_classes + 3 : num_classes + 6]
    orientations = scenes[
        :, :, num_classes + 6 : num_classes + 8
    ]  # [cos_theta, sin_theta]

    # Descale to world coordinates
    if not parse_only:
        positions = descale_pos(positions_normalized, device=device)
        sizes = descale_size(sizes_normalized, device=device)
    else:
        positions = positions_normalized
        sizes = sizes_normalized

    # Get object categories
    object_indices = torch.argmax(one_hot, dim=-1)

    # Identify empty slots
    empty_class_idx = num_classes - 1
    is_empty = object_indices == empty_class_idx

    return {
        "one_hot": one_hot,
        "positions": positions,
        "sizes": sizes,
        "orientations": orientations,
        "object_indices": object_indices,
        "is_empty": is_empty,
        "device": device,
    }


def get_all_universal_reward_functions():
    """
    Returns a dictionary of all universal (hand-designed) reward functions.

    This is a centralized place to define which universal rewards exist,
    so they don't need to be listed in multiple places.

    Returns:
        Dict mapping reward names to reward functions
    """
    # Import here to avoid circular imports
    from universal_constraint_rewards.must_have_furniture_reward import (
        compute_must_have_furniture_reward,
    )
    from universal_constraint_rewards.non_penetration_reward import (
        compute_non_penetration_reward,
    )
    from universal_constraint_rewards.object_count_reward import (
        compute_object_count_reward,
    )
    
    from universal_constraint_rewards.not_out_of_bound_reward import (
        compute_boundary_violation_reward,
    )
    
    from universal_constraint_rewards.gravity_following_reward import (
        compute_gravity_following_reward,
    )
    
    from universal_constraint_rewards.accessibility_reward import (
        compute_accessibility_reward,
    )
    
    from universal_constraint_rewards.night_tables_on_head_side_reward import (
        compute_nightstand_placement_reward,
    )
    
    from universal_constraint_rewards.axis_alignment_reward import (
        compute_axis_alignment_reward,
    )
    
    from universal_constraint_rewards.furniture_against_wall_reward import (
        compute_wall_proximity_reward,
    )
    

    return {
        "must_have_furniture": compute_must_have_furniture_reward,
        "non_penetration": compute_non_penetration_reward,
        "object_count": compute_object_count_reward,
        "not_out_of_bound": compute_boundary_violation_reward,
        "accessibility": compute_accessibility_reward,
        "gravity_following": compute_gravity_following_reward,
        "night_tables_on_head_side": compute_nightstand_placement_reward,
        "axis_alignment": compute_axis_alignment_reward,
        "furniture_against_wall": compute_wall_proximity_reward,
        
    }


def get_universal_reward(
    parsed_scene,
    reward_normalizer,
    num_classes=22,
    universal_importance_weights=None,
    get_reward_functions=None,
    **kwargs,
):
    """
    Entry point for computing universal reward from multiple reward functions.

    This function computes predefined universal reward functions and combines them.

    Args:
        parsed_scene: Dict returned by parse_and_descale_scenes()
        num_classes: Number of object classes (default: 22)
        universal_importance_weights: Dict mapping reward names to importance weights
        reward_normalizer:  normalizer to scale rewards to [0, 1] range
        get_reward_functions: Dict of reward functions to compute (if None, uses defaults)
        **kwargs: Additional arguments passed to individual reward functions

    Returns:
        total_reward: Combined reward normalized by sum of importance weights
        reward_components: Dict with individual reward values for analysis
    """
    rewards = {}

    # Define default universal reward functions if not provided
    if get_reward_functions is None:
        get_reward_functions = get_all_universal_reward_functions()

    # Compute rewards for each function
    for key, value in get_reward_functions.items():
        reward = value(parsed_scene, **kwargs)
        rewards[key] = reward

        print(f"[Ashok] Raw reward for {key}: {reward}")
        rewards[key] = reward
    # Normalize rewards if normalizer is provided
    reward_normalizer = None
    reward_components = {}
    if reward_normalizer is not None:
        for key, value in rewards.items():
            reward_components[key] = value # viz raw values to avoid weird normalized values in curves
            rewards[key] = reward_normalizer.normalize(key, torch.tensor(value))
    else:
        for key, value in rewards.items():
            reward_components[key] = value
    # Set default importance weights
    if universal_importance_weights is None:
        universal_importance_weights = {key: 1.0 for key in rewards.keys()}
    # print(f"[Ashok] importance weights: {universal_importance_weights}")
    # Combine rewards with importance weights
    rewards_sum = 0

    for key, value in rewards.items():
        importance = universal_importance_weights.get(key, 1.0)
        rewards_sum += importance * value
        

    return rewards_sum / sum(universal_importance_weights.values()), reward_components
