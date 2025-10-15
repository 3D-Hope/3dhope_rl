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


def get_universal_reward(
    parsed_scene, num_classes=22, reward_weights=None, importance_weights=None, **kwargs
):
    """
    Entry point for computing composite reward from multiple reward functions.

    This function handles parsing and descaling, then calls individual reward
    functions and combines them with TWO-STEP weighting:

    1. NORMALIZATION (technical): Scale all rewards to [-1, 0] range
    2. IMPORTANCE WEIGHTING (design): Apply domain-specific priorities

    This separation makes weights interpretable: importance_weight=1.5 means
    "50% more important", regardless of the underlying reward scales.

    NORMALIZATION FACTORS (using hybrid strategy):
        - Unbounded rewards (gravity, penetration): Tanh normalization
          → gravity: scale=1.0 (1m² floating → -0.76 normalized)
          → penetration: scale=5.0 (moderate overlap → -0.76 normalized)
        - Bounded rewards (must_have, count): Linear normalization
          → must_have: max=10.0 (true maximum)
          → object_count: max=8.0 (approximate range)

    RECOMMENDED IMPORTANCE WEIGHTS (interior design perspective):
        importance_weights = {
            'must_have_furniture': 1.5,  # Highest: functional requirement
            'gravity': 1.0,              # Critical: physics constraint
            'non_penetration': 1.0,      # Critical: physics constraint
            'object_count': 0.7,         # Important: aesthetic/realism
        }

    Note: Can specify either 'reward_weights' (legacy, direct weights) OR
    'importance_weights' (recommended, after normalization).

    Args:
        scenes: Tensor of shape (B, N, 30) where:
            - B: batch size
            - N: number of furniture slots (12)
            - 30: one_hot(22) + position(3) + size(3) + cos_theta(1) + sin_theta(1)
        num_classes: Number of object classes (default: 22, includes empty class)
        reward_weights: (LEGACY) Dict mapping reward names to direct weights.
            If provided, importance_weights is ignored.
        importance_weights: (RECOMMENDED) Dict mapping reward names to importance
            multipliers (1.0 = baseline). Rewards are normalized to [-1,0] first.
            Only specify weights you want to change from defaults.
        **kwargs: Additional arguments passed to individual reward functions
            - room_type: 'bedroom', 'living_room', etc. (for must_have_furniture)

    Returns:
        total_reward: Tensor of shape (B,) with combined rewards for each scene
        reward_components: Dict with individual reward values for analysis
    """

    # Normalization configuration (different strategies for different reward types)
    # Unbounded rewards use tanh normalization, bounded use linear
    NORMALIZATION_CONFIG = {
        "gravity": {
            "type": "tanh",
            "scale": 0.2,  # tried with 1, reward got minimized because values were in few cms,   tried with 0.01 too harsh did not learn, tanh saturated# Sensitivity
            # tried 0.1 gravity raw rewards are values in rane [-2.5  , -0.0139] mean -0.129
            # Sensitivity: 1 cm off the ground(total violations) is severely bad
            # NOT a maximum! Can handle arbitrarily large penalties smoothly
        },
        "non_penetration": {
            "type": "tanh",
            "scale": 5,  # IMP all scenes' rewards are max, no penetration at all in train and val # 5.0 led to about 10% hoping to lower it further,  (after convergence of 5.0 scale curriculum learning may be)tried with 0.05 too easy(all reward 1) did not learn, tanh saturated# Sensitivity: 5cm total penetration(considered "severe")
            # NOT a maximum! Can handle arbitrarily large overlaps smoothly
        },
        "must_have_furniture": {
            "type": "linear",
            "max": 10.0,  # True maximum: -10 (missing required furniture)
        },
        "object_count": {
            "type": "linear_shifted",
            "min": -9.2,  # Worst case: 1 object (very rare, log(0.01%) ≈ -9.2)
            "max": -1.2,  # Best case: 5 objects (most common, log(30%) ≈ -1.2)
            # Range of 8.0, will normalize to [-1, 0] where:
            # 5 objects (-1.2) → 0.0 (perfect)
            # 1 object (-9.2) → -1.0 (worst)
        },
    }

    def normalize_reward(reward, reward_name):
        """
        Normalize reward to approximately [-1, 0] range.

        Uses different strategies based on reward characteristics:
        - Tanh: For unbounded rewards (gravity, penetration)
          → scale parameter sets "severe violation" threshold
        - Linear: For bounded rewards (must_have)
          → uses true maximum
        - Linear-shifted: For bounded with offset (object_count)
          → maps known range to [-1, 0]

        Args:
            reward: Raw reward tensor
            reward_name: Name of the reward

        Returns:
            Normalized reward in approximately [-1, 0] range
        """
        config = NORMALIZATION_CONFIG[reward_name]

        if config["type"] == "tanh":
            # Soft normalization: -tanh(-reward / scale)
            # Maps unbounded penalties to [-1, 0] smoothly
            # scale parameter: when |reward| = scale → normalized ≈ -0.76
            scale = config["scale"]
            return -torch.tanh(-reward / scale)

        elif config["type"] == "linear":
            # Linear normalization for bounded rewards
            max_penalty = config["max"]
            # Clamp to prevent going beyond [-1, 0]
            clamped = torch.clamp(reward, min=-max_penalty, max=0.0)
            return clamped / max_penalty

        elif config["type"] == "linear_shifted":
            # Linear normalization with shift for bounded rewards with non-zero best value
            # Maps [min, max] → [-1, 0]
            min_val = config["min"]  # Worst value (e.g., -9.2 for 1 object)
            max_val = config["max"]  # Best value (e.g., -1.2 for 5 objects)
            range_val = max_val - min_val  # e.g., 8.0

            # Shift so max_val → 0, min_val → -1
            # normalized = (reward - max_val) / range_val
            # When reward = max_val: (max_val - max_val) / range = 0
            # When reward = min_val: (min_val - max_val) / range = -range/range = -1
            clamped = torch.clamp(reward, min=min_val, max=max_val)
            return (clamped - max_val) / range_val

        else:
            raise ValueError(f"Unknown normalization type: {config['type']}")

    # Default importance weights (interior design priorities)
    DEFAULT_IMPORTANCE = {
        "must_have_furniture": 1.5,  # Highest: room must be functional
        "gravity": 1.0,  # Critical: no floating furniture
        "non_penetration": 1.0,  # Critical: no overlapping objects
        "object_count": 0.7,  # Important: realistic clutter level
    }

    # Determine which weighting scheme to use
    if reward_weights is not None:
        # Legacy mode: use direct weights (no normalization)
        final_weights = reward_weights
        use_normalization = False
    else:
        # Recommended mode: normalize then apply importance
        if importance_weights is None:
            importance_weights = {}

        # Merge user importance with defaults
        final_importance = DEFAULT_IMPORTANCE.copy()
        final_importance.update(importance_weights)

        # For new mode, we'll normalize in the combination step
        use_normalization = True

    # Parse and descale scenes once
    # parsed_scene = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # Initialize reward components
    reward_components = {}

    # Determine which rewards to compute
    if use_normalization:
        # New mode: compute all rewards that have importance weights
        rewards_to_compute = final_importance.keys()
    else:
        # Legacy mode: compute rewards that have direct weights
        # rewards_to_compute = reward_weights.keys()
        raise

    # Compute individual rewards (import here to avoid circular import)
    if "gravity" in rewards_to_compute:
        from universal_constraint_rewards.gravity_following_reward import (
            compute_gravity_following_reward,
        )

        average_gravity = compute_gravity_following_reward(parsed_scene) / len(
            parsed_scene
        )
        print(f"average gravity reward {average_gravity}")
    #     reward_components["gravity"] = compute_gravity_following_reward(parsed_scene) #NOTE: flux diffusion baseline already has about 6mm voilaiton in average. so trying to optimize this further makes it unlearn instead, so just logging the values

    if "object_count" in rewards_to_compute:
        from universal_constraint_rewards.object_count_reward import (
            compute_object_count_reward,
        )

        # Use NLL mode for per-scene credit assignment (not KL mode)
        reward_components["object_count"] = compute_object_count_reward(
            parsed_scene, mode="nll"
        )

    if "must_have_furniture" in rewards_to_compute:
        from universal_constraint_rewards.must_have_furniture_reward import (
            compute_must_have_furniture_reward,
        )

        room_type = kwargs.get("room_type", "bedroom")
        reward_components["must_have_furniture"] = compute_must_have_furniture_reward(
            parsed_scene, room_type=room_type
        )

    if "non_penetration" in rewards_to_compute:
        from universal_constraint_rewards.non_penetration_reward import (
            compute_non_penetration_reward,
        )

        reward_components["non_penetration"] = compute_non_penetration_reward(
            parsed_scene
        )

    # Combine rewards with weights
    batch_size = parsed_scene["is_empty"].shape[0]
    device = parsed_scene["is_empty"].device
    total_reward = torch.zeros(batch_size, device=device)

    if use_normalization:
        # New mode: normalize then weight by importance
        for reward_name, raw_reward in reward_components.items():
            normalized = normalize_reward(raw_reward, reward_name)
            normalized += 1  # Shift to [0, 1] from [-1, 0]
            importance = final_importance.get(reward_name, 1.0)
            total_reward += normalized * importance
            reward_components[reward_name] = normalized
    else:
        # Legacy mode: direct weighting (no normalization)
        # for reward_name, reward_value in reward_components.items():
        #     weight = reward_weights.get(reward_name, 0.0)
        #     total_reward += weight * reward_value
        raise
    importance_sum = sum(final_importance.values())
    return (
        total_reward / importance_sum,
        reward_components,
    )  # total reward scale  [0, sum of importance weights] to [0, 1]
