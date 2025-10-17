import torch


def get_reward(parsed_scene, ideal=3.0, sigma=1.0, **kwargs):
    """
    Gaussian-shaped reward for ideal bed–TV distance.
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    positions = parsed_scene["positions"]
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
            
            if not isinstance(valid_indices, torch.Tensor):
                continue
            
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
                valid_pos = valid_pos.unsqueeze(0)
            
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

            dist = torch.norm(tv_pos - bed_pos)
            rewards[b] = torch.exp(-((dist - ideal) ** 2) / (2 * sigma**2))
            
        except Exception as e:
            print(f"[ERROR] reward_tv_distance batch {b}: {e}")
            continue

    return rewards

def test_reward():
    """
    Test function for the reward.
    Quick validation test for reward_tv_distance.
    
    For comprehensive testing, run:
        python dynamic_constraint_rewards/test_all_dynamic_rewards.py
    """
    print("Testing reward_tv_distance...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_labels = {8: "double_bed", 19: "tv_stand", 21: "empty"}
    max_objects = 12
    
    # Test 1: Ideal distance (3m)
    parsed_scene_ideal = {
        "positions": torch.tensor([
            [[0.0, 0.5, 0.0], [3.0, 0.5, 0.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 2)
        ], device=device, dtype=torch.float32),
        "object_indices": torch.tensor([[8, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long),
        "is_empty": torch.tensor([[False, False] + [True] * (max_objects - 2)], device=device, dtype=torch.bool),
        "device": device
    }
    
    reward = get_reward(parsed_scene_ideal, idx_to_labels=idx_to_labels)
    print(f"  ✓ Ideal distance (3m): {reward[0].item():.4f}")
    assert reward[0].item() > 0.9, f"Expected >0.9 for ideal distance, got {reward[0].item()}"
    
    # Test 2: Too close (1m)
    parsed_scene_close = {
        "positions": torch.tensor([
            [[0.0, 0.5, 0.0], [1.0, 0.5, 0.0]] + [[0.0, 0.0, 0.0]] * (max_objects - 2)
        ], device=device, dtype=torch.float32),
        "object_indices": torch.tensor([[8, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long),
        "is_empty": torch.tensor([[False, False] + [True] * (max_objects - 2)], device=device, dtype=torch.bool),
        "device": device
    }
    
    reward = get_reward(parsed_scene_close, idx_to_labels=idx_to_labels)
    print(f"  ✓ Too close (1m): {reward[0].item():.4f}")
    assert reward[0].item() < 0.5, f"Expected <0.5 for close distance, got {reward[0].item()}"
    
    print("✓ All tests passed!")
    print("For comprehensive testing, run: python dynamic_constraint_rewards/test_all_dynamic_rewards.py")