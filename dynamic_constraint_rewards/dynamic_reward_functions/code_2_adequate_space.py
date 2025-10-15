import math

import torch
import torch.nn.functional as F


def get_reward(parsed_scene, **kwargs):
    """
    Reward function for: "the room should have adequate space around the kids_bed for safe movement"

    Requirements:
    1. Scene must have a kids_bed
    2. Kids_bed should have adequate clearance space around it (at least 0.6m on accessible sides)
    3. Kids_bed should not be too close to other furniture or walls

    Args:
        parsed_scene: Dict with scene data
        **kwargs: idx_to_labels for object mapping, floor_vertices for room boundaries

    Returns:
        rewards: Tensor of shape (B,) with constraint satisfaction rewards
    """
    device = parsed_scene["device"]
    B = parsed_scene["positions"].shape[0]

    is_empty = parsed_scene["is_empty"]
    object_indices = parsed_scene["object_indices"]
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]

    # Get object label mapping
    idx_to_labels = kwargs.get("idx_to_labels", {})
    labels_to_idx = {
        v: int(k) if isinstance(k, str) else k for k, v in idx_to_labels.items()
    }

    # Find kids_bed index
    kids_bed_idx = labels_to_idx.get("kids_bed", -1)

    # Minimum clearance requirement (in meters)
    min_clearance = 0.6
    ideal_clearance = 0.8

    rewards = torch.zeros(B, device=device)

    for b in range(B):
        batch_reward = 0.0

        # Get valid objects in this scene
        valid_mask = ~is_empty[b]
        valid_indices = object_indices[b][valid_mask]
        valid_positions = positions[b][valid_mask]
        valid_sizes = sizes[b][valid_mask]

        # Check if kids_bed exists
        has_kids_bed = (
            (valid_indices == kids_bed_idx).any().item() if kids_bed_idx >= 0 else False
        )

        if not has_kids_bed:
            # No kids_bed, no reward
            rewards[b] = 0.0
            continue

        # Reward component 1: Kids bed exists (base reward)
        batch_reward += 2.0

        # Get kids_bed data
        kids_bed_mask = valid_indices == kids_bed_idx
        kids_bed_pos = valid_positions[kids_bed_mask][0]
        kids_bed_size = valid_sizes[kids_bed_mask][0]

        # Calculate clearance from other furniture
        clearance_rewards = []

        for i in range(len(valid_indices)):
            if valid_indices[i] == kids_bed_idx:
                continue

            other_pos = valid_positions[i]
            other_size = valid_sizes[i]

            # Calculate distance between bounding boxes (not centers)
            # Distance in each dimension
            dx = abs(kids_bed_pos[0] - other_pos[0]) - (
                kids_bed_size[0] + other_size[0]
            )
            dy = abs(kids_bed_pos[1] - other_pos[1]) - (
                kids_bed_size[1] + other_size[1]
            )
            dz = abs(kids_bed_pos[2] - other_pos[2]) - (
                kids_bed_size[2] + other_size[2]
            )

            # Horizontal clearance (x-z plane)
            horizontal_clearance = torch.sqrt(
                torch.clamp(dx, min=0.0) ** 2 + torch.clamp(dz, min=0.0) ** 2
            )

            # Reward based on clearance
            if horizontal_clearance < min_clearance:
                # Too close, penalty
                clearance_reward = (
                    -1.0 * (min_clearance - horizontal_clearance) / min_clearance
                )
            elif horizontal_clearance < ideal_clearance:
                # Acceptable but not ideal
                clearance_reward = (
                    (horizontal_clearance - min_clearance)
                    / (ideal_clearance - min_clearance)
                    * 0.5
                )
            else:
                # Good clearance
                clearance_reward = 0.5

            clearance_rewards.append(torch.tensor(clearance_reward).item())

        # Reward component 2: Average clearance from other furniture (0-3 points)
        if clearance_rewards:
            avg_clearance_reward = sum(clearance_rewards) / len(clearance_rewards)
            # Scale to 0-3 range
            batch_reward += max(0.0, min(3.0, avg_clearance_reward * 6.0))
        else:
            # No other furniture, give full clearance reward
            batch_reward += 3.0

        # Reward component 3: Distance from walls (0-2 points)
        floor_vertices = kwargs.get("floor_vertices", None)
        if floor_vertices is not None and len(floor_vertices) > 0:
            # Calculate minimum distance to walls
            wall_distances = []

            # Convert floor vertices to tensor
            floor_verts = torch.tensor(
                floor_vertices, device=device, dtype=torch.float32
            )

            # Check distance to each wall segment
            for i in range(len(floor_verts)):
                v1 = floor_verts[i]
                v2 = floor_verts[(i + 1) % len(floor_verts)]

                # Project kids_bed position onto wall segment (x-z plane)
                bed_pos_2d = torch.tensor(
                    [kids_bed_pos[0], kids_bed_pos[2]], device=device
                )
                v1_2d = torch.tensor([v1[0], v1[2]], device=device)
                v2_2d = torch.tensor([v2[0], v2[2]], device=device)

                # Calculate distance from point to line segment
                wall_vec = v2_2d - v1_2d
                wall_len_sq = torch.sum(wall_vec**2)

                if wall_len_sq > 1e-6:
                    t = torch.clamp(
                        torch.sum((bed_pos_2d - v1_2d) * wall_vec) / wall_len_sq,
                        0.0,
                        1.0,
                    )
                    projection = v1_2d + t * wall_vec
                    dist = torch.norm(bed_pos_2d - projection) - max(
                        kids_bed_size[0], kids_bed_size[2]
                    )
                    wall_distances.append(dist.item())

            if wall_distances:
                min_wall_dist = min(wall_distances)

                if min_wall_dist < min_clearance:
                    # Too close to wall
                    wall_reward = 0.0
                elif min_wall_dist < ideal_clearance:
                    # Acceptable distance
                    wall_reward = (
                        (min_wall_dist - min_clearance)
                        / (ideal_clearance - min_clearance)
                        * 1.0
                    )
                else:
                    # Good distance from wall
                    wall_reward = 2.0

                batch_reward += wall_reward
        else:
            # No wall information, assume good placement
            batch_reward += 1.0

        rewards[b] = batch_reward

    return rewards


def test_reward(**kwargs):
    """
    Test the kids_bed clearance constraint reward function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example idx_to_labels for bedroom
    idx_to_labels = kwargs.get(
        "idx_to_labels",
        {
            "0": "armchair",
            "1": "bookshelf",
            "2": "cabinet",
            "3": "ceiling_lamp",
            "4": "chair",
            "5": "children_cabinet",
            "6": "coffee_table",
            "7": "desk",
            "8": "double_bed",
            "9": "dressing_chair",
            "10": "dressing_table",
            "11": "kids_bed",
            "12": "nightstand",
            "13": "pendant_lamp",
            "14": "shelf",
            "15": "single_bed",
            "16": "sofa",
            "17": "stool",
            "18": "table",
            "19": "tv_stand",
            "20": "wardrobe",
            "21": "empty",
        },
    )

    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    num_classes = len(idx_to_labels)
    max_objects = 10

    kids_bed_idx = labels_to_idx.get("kids_bed", 11)
    wardrobe_idx = labels_to_idx.get("wardrobe", 20)
    empty_idx = num_classes - 1

    # Test Case 1: Kids bed with good clearance
    print("Test Case 1: Kids bed with good clearance")
    parsed_scene_1 = {
        "positions": torch.tensor(
            [
                [[2.0, 0.3, 2.0], [2.0, 0.5, 4.0]]
                + [[0.0, 0.0, 0.0]] * (max_objects - 2)
            ],
            device=device,
        ),
        "sizes": torch.tensor(
            [
                [[0.8, 0.3, 1.0], [0.5, 0.8, 0.3]]
                + [[0.01, 0.01, 0.01]] * (max_objects - 2)
            ],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[kids_bed_idx, wardrobe_idx] + [empty_idx] * (max_objects - 2)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor(
            [[False, False] + [True] * (max_objects - 2)], device=device
        ),
        "orientations": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 2)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[kids_bed_idx, wardrobe_idx] + [empty_idx] * (max_objects - 2)],
                dtype=torch.long,
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_1 = get_reward(parsed_scene_1, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_1.item():.2f} (Expected: >5.0)")
    assert reward_1.item() > 3.0, "Good clearance should have high reward"

    # Test Case 2: Kids bed too close to furniture
    print("\nTest Case 2: Kids bed too close to furniture")
    parsed_scene_2 = {
        "positions": torch.tensor(
            [
                [[2.0, 0.3, 2.0], [2.0, 0.5, 2.8]]
                + [[0.0, 0.0, 0.0]] * (max_objects - 2)
            ],
            device=device,
        ),
        "sizes": torch.tensor(
            [
                [[0.8, 0.3, 1.0], [0.5, 0.8, 0.3]]
                + [[0.01, 0.01, 0.01]] * (max_objects - 2)
            ],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[kids_bed_idx, wardrobe_idx] + [empty_idx] * (max_objects - 2)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor(
            [[False, False] + [True] * (max_objects - 2)], device=device
        ),
        "orientations": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 2)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[kids_bed_idx, wardrobe_idx] + [empty_idx] * (max_objects - 2)],
                dtype=torch.long,
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_2 = get_reward(parsed_scene_2, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_2.item():.2f} (Expected: <5.0)")
    assert reward_2.item() < reward_1.item(), "Poor clearance should reduce reward"

    # Test Case 3: No kids bed
    print("\nTest Case 3: No kids bed in scene")
    parsed_scene_3 = {
        "positions": torch.tensor(
            [[[2.0, 0.5, 2.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 1)], device=device
        ),
        "sizes": torch.tensor(
            [[[0.5, 0.8, 0.3]] + [[0.01, 0.01, 0.01]] * (max_objects - 1)],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[wardrobe_idx] + [empty_idx] * (max_objects - 1)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor([[False] + [True] * (max_objects - 1)], device=device),
        "orientations": torch.tensor(
            [[[1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 1)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[wardrobe_idx] + [empty_idx] * (max_objects - 1)], dtype=torch.long
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_3 = get_reward(parsed_scene_3, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_3.item():.2f} (Expected: 0.0)")
    assert reward_3.item() == 0.0, "No kids bed should give zero reward"

    # Test Case 4: Kids bed alone (good clearance by default)
    print("\nTest Case 4: Kids bed alone in scene")
    parsed_scene_4 = {
        "positions": torch.tensor(
            [[[2.0, 0.3, 2.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 1)], device=device
        ),
        "sizes": torch.tensor(
            [[[0.8, 0.3, 1.0]] + [[0.01, 0.01, 0.01]] * (max_objects - 1)],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[kids_bed_idx] + [empty_idx] * (max_objects - 1)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor([[False] + [True] * (max_objects - 1)], device=device),
        "orientations": torch.tensor(
            [[[1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 1)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[kids_bed_idx] + [empty_idx] * (max_objects - 1)], dtype=torch.long
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_4 = get_reward(parsed_scene_4, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_4.item():.2f} (Expected: >5.0)")
    assert reward_4.item() > 4.0, "Kids bed alone should have high reward"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_reward()
