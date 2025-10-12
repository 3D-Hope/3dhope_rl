import numpy as np
import torch

from physical_constraint_rewards.commons import ceiling_objects, idx_to_labels


def compute_gravity_following_reward(parsed_scene):
    """
    Calculate gravity-following reward based on how close objects are to the ground.

    Objects should rest on the floor (y_min ≈ 0), except for ceiling objects like lamps.

    Args:
        parsed_scene: Dict returned by parse_and_descale_scenes()

    Returns:
        rewards: Tensor of shape (B,) with gravity-following rewards for each scene
    """
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]

    # Identify ceiling objects
    ceiling_indices = [
        idx for idx, label in idx_to_labels.items() if label in ceiling_objects
    ]
    is_ceiling = torch.zeros_like(is_empty, dtype=torch.bool)
    for ceiling_idx in ceiling_indices:
        is_ceiling |= object_indices == ceiling_idx

    # Create mask for objects that should follow gravity (non-empty, non-ceiling)
    should_follow_gravity = ~is_empty & ~is_ceiling

    # Calculate y_min for each object (bottom of bounding box)
    # y_min = center_y - half_extent_y
    # NOTE: sizes are already half-extents (sx/2, sy/2, sz/2), so no need to divide by 2
    y_centers = positions[:, :, 1]  # (B, N)
    y_half_extents = sizes[:, :, 1]  # (B, N) - already half-extents
    y_min = y_centers - y_half_extents  # (B, N)

    # Calculate distance from floor (should be ~0 for objects on the ground)
    # Use absolute value to penalize both floating and sinking objects
    floor_distance = torch.abs(y_min)  # (B, N)

    # Apply mask to only consider relevant objects
    masked_distances = torch.where(
        should_follow_gravity, floor_distance, torch.zeros_like(floor_distance)
    )

    # Sum distances per scene
    total_distance = masked_distances.sum(
        dim=1
    )  # (B,)(using this as penalty: enalizes scenes with more objects more heavily, regardless of quality. so use avg)

    # Count number of gravity-following objects to normalize
    # num_gravity_objects = should_follow_gravity.sum(dim=1).float()  # (B,)

    # # Avoid division by zero for scenes with no gravity-following objects
    # # If all objects are ceiling or empty, give neutral reward (0)
    # avg_distance = torch.where(
    #     num_gravity_objects > 0,
    #     total_distance / num_gravity_objects,
    #     torch.zeros_like(total_distance)
    # )

    # # Convert distance to reward (closer to ground = higher reward)
    # reward = -avg_distance
    reward = (
        -total_distance #the values are in meters, this value is very small because only few mm off the ground was observed in most cases;
    )  # using total distance as penalty: penalizes scenes with more objects more heavily, regardless of quality.
    return reward


def test_gravity_following_reward():
    """Test cases for gravity following reward."""
    print("\n" + "=" * 60)
    print("Testing Gravity Following Reward")
    print("=" * 60)

    device = "cpu"
    num_classes = 22
    batch_size = 3
    num_objects = 12

    # Helper to create a scene tensor
    def create_scene(object_types, positions_normalized, sizes_normalized):
        """
        object_types: list of object indices (length N)
        positions_normalized: list of [x, y, z] in normalized coords (length N)
        sizes_normalized: list of [x, y, z] in normalized coords (length N)
        """
        scene = torch.zeros(num_objects, 30, device=device)

        for i, (obj_type, pos, size) in enumerate(
            zip(object_types, positions_normalized, sizes_normalized)
        ):
            # One-hot encoding
            scene[i, obj_type] = 1.0
            # Position (normalized -1 to 1)
            scene[i, num_classes : num_classes + 3] = torch.tensor(pos, device=device)
            # Size (normalized -1 to 1)
            scene[i, num_classes + 3 : num_classes + 6] = torch.tensor(
                size, device=device
            )
            # Orientation (cos, sin)
            scene[i, num_classes + 6 : num_classes + 8] = torch.tensor(
                [1.0, 0.0], device=device
            )

        return scene

    # Test Case 1: Perfect scene - all objects on ground
    print("\nTest 1: All objects perfectly on ground")
    # Chair at y=0 (normalized y=-1 gives actual y≈0.045)
    # Use y=-0.95 to get closer to y=0
    scene1 = create_scene(
        object_types=[
            4,
            4,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
        ],  # 2 chairs, rest empty
        positions_normalized=[
            [0.0, -0.98, 0.0],  # Chair 1 - very close to ground
            [0.5, -0.98, 0.5],  # Chair 2 - very close to ground
        ]
        + [[0.0, 0.0, 0.0]] * 10,
        sizes_normalized=[
            [0.0, 0.0, 0.0],  # Small chair
            [0.0, 0.0, 0.0],  # Small chair
        ]
        + [[0.0, 0.0, 0.0]] * 10,
    )

    # Test Case 2: Objects floating above ground
    print("Test 2: Objects floating above ground")
    scene2 = create_scene(
        object_types=[
            4,
            16,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
        ],  # chair, sofa, rest empty
        positions_normalized=[
            [0.0, 0.5, 0.0],  # Chair floating high
            [0.5, 0.3, 0.5],  # Sofa floating
        ]
        + [[0.0, 0.0, 0.0]] * 10,
        sizes_normalized=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        + [[0.0, 0.0, 0.0]] * 10,
    )

    # Test Case 3: Mix of ceiling objects and ground objects
    print("Test 3: Ceiling lamp (ignored) + chair on ground")
    scene3 = create_scene(
        object_types=[
            3,
            4,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
            21,
        ],  # ceiling_lamp, chair, rest empty
        positions_normalized=[
            [0.0, 0.8, 0.0],  # Ceiling lamp (should be ignored)
            [0.5, -0.98, 0.5],  # Chair on ground
        ]
        + [[0.0, 0.0, 0.0]] * 10,
        sizes_normalized=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
        + [[0.0, 0.0, 0.0]] * 10,
    )

    # Batch the scenes
    scenes = torch.stack([scene1, scene2, scene3], dim=0)

    # Import parse function
    from physical_constraint_rewards.commons import parse_and_descale_scenes

    # Parse scenes
    parsed = parse_and_descale_scenes(scenes, num_classes=num_classes)

    # Compute rewards
    rewards = compute_gravity_following_reward(parsed)

    print(f"\nResults:")
    print(
        f"Scene 1 (objects on ground): {rewards[0].item():.4f} (should be close to 0)"
    )
    print(f"Scene 2 (floating objects): {rewards[1].item():.4f} (should be negative)")
    print(
        f"Scene 3 (ceiling lamp ignored): {rewards[2].item():.4f} (should be close to 0)"
    )

    # Verify
    assert (
        rewards[0] > rewards[1]
    ), "Scene with grounded objects should have higher reward than floating"
    assert rewards[0] > -0.5, "Grounded scene should have reward close to 0"
    assert rewards[1] < -0.5, "Floating scene should have significantly negative reward"

    print("\n✓ All gravity tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_gravity_following_reward()
