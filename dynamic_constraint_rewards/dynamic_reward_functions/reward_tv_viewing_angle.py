import torch
import torch.nn.functional as F


def get_reward(parsed_scene, **kwargs):
    """
    Reward for bed/sofa facing the TV.
    Uses cosine similarity between their facing directions.
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    positions = parsed_scene["positions"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    idx_to_labels = kwargs["idx_to_labels"]
    
    # Handle both integer and string keys
    if idx_to_labels and isinstance(list(idx_to_labels.keys())[0], str):
        idx_to_labels = {int(k): v for k, v in idx_to_labels.items()}
    
    idx_tv = next((k for k, v in idx_to_labels.items() if "tv_stand" in v), None)
    idx_bed = next((k for k, v in idx_to_labels.items() if "bed" in v or "sofa" in v), None)
    if idx_tv is None or idx_bed is None:
        return torch.zeros(len(object_indices), device=device)

    rewards = torch.zeros(len(object_indices), device=device)
    for b in range(len(object_indices)):
        try:
            # Get valid mask - ensure boolean tensor
            if isinstance(is_empty, torch.Tensor):
                valid_mask = ~is_empty[b]
            else:
                valid_mask = ~torch.tensor(is_empty[b], dtype=torch.bool, device=device)
            
            # Convert to boolean explicitly
            if isinstance(valid_mask, torch.Tensor):
                if valid_mask.dtype != torch.bool:
                    valid_mask = valid_mask.bool()
            else:
                # valid_mask is a Python bool - no valid objects
                continue
            
            # Check if we have any valid objects
            if valid_mask.sum().item() == 0:
                continue
                
            valid_indices = object_indices[b][valid_mask]
            valid_pos = positions[b][valid_mask]
            valid_orient = orientations[b][valid_mask]
            
            if not isinstance(valid_indices, torch.Tensor):
                continue
            
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
                valid_pos = valid_pos.unsqueeze(0)
                valid_orient = valid_orient.unsqueeze(0)
            
            if valid_indices.numel() == 0:
                continue

            # Check for TV and bed - ensure tensor comparisons
            tv_mask = (valid_indices == idx_tv)
            bed_mask = (valid_indices == idx_bed)
            
            if isinstance(tv_mask, torch.Tensor):
                has_tv = tv_mask.any().item()
            else:
                has_tv = bool(tv_mask)
                
            if isinstance(bed_mask, torch.Tensor):
                has_bed = bed_mask.any().item()
            else:
                has_bed = bool(bed_mask)
            
            if not (has_tv and has_bed):
                continue

            tv_pos = valid_pos[valid_indices == idx_tv][0]
            bed_pos = valid_pos[valid_indices == idx_bed][0]
            bed_dir = valid_orient[valid_indices == idx_bed][0]

            # Compute direction from bed to TV (in XZ plane, ignore Y)
            dir_bed_to_tv = tv_pos - bed_pos
            # Project to 2D (XZ plane) to match orientation which is [cos, sin] in XZ
            dir_bed_to_tv_2d = torch.tensor([dir_bed_to_tv[0], dir_bed_to_tv[2]], device=device)
            dir_bed_to_tv_2d = dir_bed_to_tv_2d / (torch.norm(dir_bed_to_tv_2d) + 1e-6)

            alignment = F.cosine_similarity(bed_dir.unsqueeze(0), dir_bed_to_tv_2d.unsqueeze(0)).clamp(0, 1)
            rewards[b] = alignment.item()
            
        except Exception as e:
            print(f"[ERROR] reward_tv_viewing_angle batch {b}: {e}")
            continue

    return rewards


def test_reward():
    """
    Test function for the reward.
    Quick validation test for reward_tv_viewing_angle.
    
    For comprehensive testing, run:
        python dynamic_constraint_rewards/test_all_dynamic_rewards.py
    """
    print("Testing reward_tv_viewing_angle...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_labels = {8: "double_bed", 19: "tv_stand", 21: "empty"}
    max_objects = 12
    
    # Test 1: Good alignment (bed facing TV)
    parsed_scene_aligned = {
        "positions": torch.tensor([
            [[0.0, 0.5, 0.0], [3.0, 0.5, 0.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 2)
        ], device=device, dtype=torch.float32),
        "orientations": torch.tensor([
            [[1.0, 0.0], [-1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 2)
        ], device=device, dtype=torch.float32),
        "object_indices": torch.tensor([[8, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long),
        "is_empty": torch.tensor([[False, False] + [True] * (max_objects - 2)], device=device, dtype=torch.bool),
        "device": device
    }
    
    reward = get_reward(parsed_scene_aligned, idx_to_labels=idx_to_labels)
    print(f"  ✓ Good alignment: {reward[0].item():.4f}")
    assert reward[0].item() > 0.9, f"Expected >0.9 for aligned, got {reward[0].item()}"
    
    # Test 2: Poor alignment (bed facing away)
    parsed_scene_misaligned = {
        "positions": torch.tensor([
            [[0.0, 0.5, 0.0], [3.0, 0.5, 0.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 2)
        ], device=device, dtype=torch.float32),
        "orientations": torch.tensor([
            [[-1.0, 0.0], [-1.0, 0.0]] + [[0.0, 0.0]] * (max_objects - 2)
        ], device=device, dtype=torch.float32),
        "object_indices": torch.tensor([[8, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long),
        "is_empty": torch.tensor([[False, False] + [True] * (max_objects - 2)], device=device, dtype=torch.bool),
        "device": device
    }
    
    reward = get_reward(parsed_scene_misaligned, idx_to_labels=idx_to_labels)
    print(f"  ✓ Poor alignment: {reward[0].item():.4f}")
    assert reward[0].item() < 0.1, f"Expected <0.1 for misaligned, got {reward[0].item()}"
    
    print("✓ All tests passed!")
    print("For comprehensive testing, run: python dynamic_constraint_rewards/test_all_dynamic_rewards.py")