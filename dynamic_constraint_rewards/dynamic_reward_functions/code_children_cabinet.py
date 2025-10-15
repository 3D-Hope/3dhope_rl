import math

import torch
import torch.nn.functional as F


def get_reward(parsed_scene, **kwargs):
    """
    Reward function for: "a children_cabinet should be included for storage"

    Requirements:
    1. Scene must have at least one children_cabinet
    2. Higher reward for having a children_cabinet present

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
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}

    # Find children_cabinet index
    children_cabinet_idx = None
    for label_key in [
        "children_cabinet",
        "children cabinet",
        "kids_cabinet",
        "kids cabinet",
    ]:
        if label_key in labels_to_idx:
            children_cabinet_idx = int(labels_to_idx[label_key])
            break

    rewards = torch.zeros(B, device=device)

    for b in range(B):
        batch_reward = 0.0

        # Get valid objects in this scene
        valid_mask = ~is_empty[b]
        valid_indices = object_indices[b][valid_mask]

        # Check for children_cabinet
        has_children_cabinet = False
        if children_cabinet_idx is not None:
            has_children_cabinet = (valid_indices == children_cabinet_idx).any().item()

        # Reward for having a children_cabinet (10 points for presence)
        if has_children_cabinet:
            batch_reward += 10.0

        rewards[b] = batch_reward

    return rewards


def test_reward(**kwargs):
    """
    Test the children_cabinet constraint reward function.
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

    labels_to_idx = {v: k for k, v in idx_to_labels.items()}

    num_classes = len(idx_to_labels)
    max_objects = 10
    B = 1

    children_cabinet_idx = int(labels_to_idx.get("children_cabinet", 5))
    kids_bed_idx = int(labels_to_idx.get("kids_bed", 11))
    empty_idx = num_classes - 1

    # Test Case 1: Scene with children_cabinet
    print("Test Case 1: Scene with children_cabinet")
    parsed_scene_1 = {
        "positions": torch.tensor(
            [
                [[1.0, 0.5, 0.0], [2.0, 0.5, 1.0], [3.0, 0.5, 2.0]]
                + [[0.0, 0.0, 0.0]] * (max_objects - 3)
            ],
            device=device,
        ),
        "sizes": torch.tensor(
            [
                [[0.5, 0.4, 0.3], [0.8, 0.4, 0.4], [0.4, 0.3, 0.3]]
                + [[0.01, 0.01, 0.01]] * (max_objects - 3)
            ],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[children_cabinet_idx, kids_bed_idx, 0] + [empty_idx] * (max_objects - 3)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor(
            [[False, False, False] + [True] * (max_objects - 3)], device=device
        ),
        "orientations": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 3)],
            device=device,
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [
                    [children_cabinet_idx, kids_bed_idx, 0]
                    + [empty_idx] * (max_objects - 3)
                ],
                dtype=torch.long,
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_1 = get_reward(parsed_scene_1, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_1.item():.2f} (Expected: 10.0)")
    assert (
        reward_1.item() == 10.0
    ), f"Scene with children_cabinet should have reward 10.0, got {reward_1.item()}"

    # Test Case 2: Scene without children_cabinet
    print("\nTest Case 2: Scene without children_cabinet")
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
                [[0.8, 0.4, 0.4], [0.4, 0.3, 0.3]]
                + [[0.01, 0.01, 0.01]] * (max_objects - 2)
            ],
            device=device,
        ),
        "object_indices": torch.tensor(
            [[kids_bed_idx, 0] + [empty_idx] * (max_objects - 2)],
            device=device,
            dtype=torch.long,
        ),
        "is_empty": torch.tensor(
            [[False, False] + [True] * (max_objects - 2)], device=device
        ),
        "orientations": torch.tensor(
            [[[0.0, 1.0], [1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 2)], device=device
        ),
        "one_hot": F.one_hot(
            torch.tensor(
                [[kids_bed_idx, 0] + [empty_idx] * (max_objects - 2)], dtype=torch.long
            ),
            num_classes,
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_2 = get_reward(parsed_scene_2, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_2.item():.2f} (Expected: 0.0)")
    assert (
        reward_2.item() == 0.0
    ), f"Scene without children_cabinet should have reward 0.0, got {reward_2.item()}"

    # Test Case 3: Empty scene
    print("\nTest Case 3: Empty scene")
    parsed_scene_3 = {
        "positions": torch.tensor([[[0.0, 0.0, 0.0]] * max_objects], device=device),
        "sizes": torch.tensor([[[0.01, 0.01, 0.01]] * max_objects], device=device),
        "object_indices": torch.tensor(
            [[empty_idx] * max_objects], device=device, dtype=torch.long
        ),
        "is_empty": torch.tensor([[True] * max_objects], device=device),
        "orientations": torch.tensor([[[0.0, 0.0]] * max_objects], device=device),
        "one_hot": F.one_hot(
            torch.tensor([[empty_idx] * max_objects], dtype=torch.long), num_classes
        )
        .float()
        .to(device),
        "device": device,
    }

    reward_3 = get_reward(parsed_scene_3, idx_to_labels=idx_to_labels)
    print(f"Reward: {reward_3.item():.2f} (Expected: 0.0)")
    assert (
        reward_3.item() == 0.0
    ), f"Empty scene should have reward 0.0, got {reward_3.item()}"

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_reward()
