import math

import torch
import torch.nn.functional as F


def get_reward(parsed_scene, **kwargs):
    """
    Reward function for: "a kids_bed must be present in the room"

    Requirements:
    1. Scene must have a kids_bed

    Args:
        parsed_scene: Dict with scene data
        **kwargs: idx_to_labels for object mapping

    Returns:
        rewards: Tensor of shape (B,) with constraint satisfaction rewards
    """
    device = parsed_scene["device"]
    B = parsed_scene["positions"].shape[0]

    is_empty = parsed_scene["is_empty"]
    object_indices = parsed_scene["object_indices"]

    # Get object label mapping
    idx_to_labels = kwargs.get("idx_to_labels", {})
    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}

    # Find kids_bed index
    kids_bed_idx = labels_to_idx.get("kids_bed", -1)

    rewards = torch.zeros(B, device=device)

    for b in range(B):
        batch_reward = 0.0

        # Get valid objects in this scene
        valid_mask = ~is_empty[b]
        valid_indices = object_indices[b][valid_mask]

        # Check for kids_bed
        has_kids_bed = (
            (valid_indices == kids_bed_idx).any().item() if kids_bed_idx >= 0 else False
        )

        # Reward for kids_bed presence
        if has_kids_bed:
            batch_reward += 1.0

        rewards[b] = batch_reward

    return rewards


def test_reward(**kwargs):
    """
    Test the kids_bed constraint reward function.
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
        },
    )

    labels_to_idx = {v: int(k) for k, v in idx_to_labels.items()}
    num_classes = len(idx_to_labels) + 1
    max_objects = 10

    kids_bed_idx = labels_to_idx.get("kids_bed", 11)
    armchair_idx = labels_to_idx.get("armchair", 0)

    # Test Case 1: Scene with kids_bed
    print("Test Case 1: Scene with kids_bed")
    parsed_scene_1 = {
        "positions": torch.tensor(
            [
                [[1.0, 0.5, 0.0], [2.0, 0.5, 1.0]]
                + [[0.0, 0.0, 0.0]] * (max_objects - 2)
            ],
            device=device,
        ),
        "sizes": torch.tensor(
            [
                [[0.9, 0.3, 0.7], [0.4, 0.4, 0.4]]
                + [[0.01, 0.01, 0.01]] * (max_objects - 2)
            ],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[kids_bed_idx, armchair_idx] + [num_classes - 1] * (max_objects - 2)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor(
            [[False, False] + [True] * (max_objects - 2)], device=device
        ),
        "orientations": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (max_objects - 2)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[kids_bed_idx, armchair_idx] + [num_classes - 1] * (max_objects - 2)],
                dtype=torch.long,
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_1 = get_reward(parsed_scene_1, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_1.item():.2f} (Expected: 1.0)")
    assert reward_1.item() == 1.0, "Scene with kids_bed should have reward of 1.0"

    # Test Case 2: Scene without kids_bed
    print("\nTest Case 2: Scene without kids_bed")
    parsed_scene_2 = {
        "positions": torch.tensor(
            [
                [[1.0, 0.5, 0.0], [2.0, 0.5, 1.0]]
                + [[0.0, 0.0, 0.0]] * (max_objects - 2)
            ],
            device=device,
        ),
        "sizes": torch.tensor(
            [
                [[0.4, 0.4, 0.4], [0.5, 0.3, 0.3]]
                + [[0.01, 0.01, 0.01]] * (max_objects - 2)
            ],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[armchair_idx, armchair_idx] + [num_classes - 1] * (max_objects - 2)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor(
            [[False, False] + [True] * (max_objects - 2)], device=device
        ),
        "orientations": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]] + [[0.0, 0.0]] * (max_objects - 2)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[armchair_idx, armchair_idx] + [num_classes - 1] * (max_objects - 2)],
                dtype=torch.long,
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_2 = get_reward(parsed_scene_2, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_2.item():.2f} (Expected: 0.0)")
    assert reward_2.item() == 0.0, "Scene without kids_bed should have reward of 0.0"

    # Test Case 3: Empty scene
    print("\nTest Case 3: Empty scene")
    parsed_scene_3 = {
        "positions": torch.tensor([[[0.0, 0.0, 0.0]] * max_objects], device=device),
        "sizes": torch.tensor([[[0.01, 0.01, 0.01]] * max_objects], device=device),
        "object_indices": torch.tensor(
            [[num_classes - 1] * max_objects], device=device, dtype=torch.long
        ),
        "is_empty": torch.tensor([[True] * max_objects], device=device),
        "orientations": torch.tensor([[[0.0, 0.0]] * max_objects], device=device),
        "one_hot": F.one_hot(
            torch.tensor([[num_classes - 1] * max_objects], dtype=torch.long),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_3 = get_reward(parsed_scene_3, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_3.item():.2f} (Expected: 0.0)")
    assert reward_3.item() == 0.0, "Empty scene should have reward of 0.0"

    # Test Case 4: Batch processing
    print("\nTest Case 4: Batch processing")
    batch_positions = torch.cat(
        [
            parsed_scene_1["positions"],
            parsed_scene_2["positions"],
            parsed_scene_3["positions"],
        ],
        dim=0,
    )
    batch_sizes = torch.cat(
        [parsed_scene_1["sizes"], parsed_scene_2["sizes"], parsed_scene_3["sizes"]],
        dim=0,
    )
    batch_indices = torch.cat(
        [
            parsed_scene_1["object_indices"],
            parsed_scene_2["object_indices"],
            parsed_scene_3["object_indices"],
        ],
        dim=0,
    )
    batch_is_empty = torch.cat(
        [
            parsed_scene_1["is_empty"],
            parsed_scene_2["is_empty"],
            parsed_scene_3["is_empty"],
        ],
        dim=0,
    )
    batch_orientations = torch.cat(
        [
            parsed_scene_1["orientations"],
            parsed_scene_2["orientations"],
            parsed_scene_3["orientations"],
        ],
        dim=0,
    )
    batch_one_hot = torch.cat(
        [
            parsed_scene_1["one_hot"],
            parsed_scene_2["one_hot"],
            parsed_scene_3["one_hot"],
        ],
        dim=0,
    )

    parsed_scene_batch = {
        "positions": batch_positions,
        "sizes": batch_sizes,
        "object_indices": batch_indices,
        "is_empty": batch_is_empty,
        "orientations": batch_orientations,
        "one_hot": batch_one_hot,
        "device": device,
    }

    rewards_batch = get_reward(parsed_scene_batch, idx_to_labels=idx_to_labels)
    print(f"Batch rewards: {rewards_batch}")
    assert rewards_batch[0].item() == 1.0, "First scene should have reward 1.0"
    assert rewards_batch[1].item() == 0.0, "Second scene should have reward 0.0"
    assert rewards_batch[2].item() == 0.0, "Third scene should have reward 0.0"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_reward()
