import torch


def get_reward(parsed_scene, **kwargs):
    """
    Reward for having a TV (or tv_stand) in the scene.
    Higher reward if present, lower if absent.
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]
    
    idx_to_labels = kwargs.get("idx_to_labels", {})
    
    # Handle both integer and string keys
    if idx_to_labels and isinstance(list(idx_to_labels.keys())[0], str):
        # Keys are strings, convert to int
        idx_to_labels = {int(k): v for k, v in idx_to_labels.items()}
    
    tv_idx = next((k for k, v in idx_to_labels.items() if "tv_stand" in v.lower()), None)
    if tv_idx is None:
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
            
            # Get valid indices
            valid_indices = object_indices[b][valid_mask]
            
            # Ensure it's a tensor
            if not isinstance(valid_indices, torch.Tensor):
                continue
            
            # Ensure it's at least 1D
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            if valid_indices.numel() == 0:
                continue
            
            # Check for TV - ensure tensor comparison
            tv_mask = (valid_indices == tv_idx)
            if isinstance(tv_mask, torch.Tensor):
                has_tv = tv_mask.any().item()
            else:
                has_tv = bool(tv_mask)
            
            rewards[b] = 1.0 if has_tv else 0.0
            
        except Exception as e:
            print(f"[ERROR] reward_tv_present batch {b}: {e}")
            import traceback
            traceback.print_exc()
            continue
    return rewards

def test_reward():
    """
    Test function for the reward.
    Quick validation test for reward_tv_present.
    
    For comprehensive testing, run:
        python dynamic_constraint_rewards/test_all_dynamic_rewards.py
    """
    import torch.nn.functional as F
    
    print("Testing reward_tv_present...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_to_labels = {8: "double_bed", 19: "tv_stand", 21: "empty"}
    num_classes = 22
    max_objects = 12
    
    # Test 1: TV present
    parsed_scene_with_tv = {
        "one_hot": F.one_hot(
            torch.tensor([[8, 19] + [21] * (max_objects - 2)], dtype=torch.long),
            num_classes
        ).float().to(device),
        "object_indices": torch.tensor([[8, 19] + [21] * (max_objects - 2)], device=device, dtype=torch.long),
        "is_empty": torch.tensor([[False, False] + [True] * (max_objects - 2)], device=device, dtype=torch.bool),
        "device": device
    }
    
    reward = get_reward(parsed_scene_with_tv, idx_to_labels=idx_to_labels)
    assert reward[0].item() == 1.0, f"Expected 1.0 for TV present, got {reward[0].item()}"
    print(f"  ✓ TV present: {reward[0].item()}")
    
    # Test 2: No TV
    parsed_scene_no_tv = {
        "one_hot": F.one_hot(
            torch.tensor([[8] + [21] * (max_objects - 1)], dtype=torch.long),
            num_classes
        ).float().to(device),
        "object_indices": torch.tensor([[8] + [21] * (max_objects - 1)], device=device, dtype=torch.long),
        "is_empty": torch.tensor([[False] + [True] * (max_objects - 1)], device=device, dtype=torch.bool),
        "device": device
    }
    
    reward = get_reward(parsed_scene_no_tv, idx_to_labels=idx_to_labels)
    assert reward[0].item() == 0.0, f"Expected 0.0 for no TV, got {reward[0].item()}"
    print(f"  ✓ No TV: {reward[0].item()}")
    
    print("✓ All tests passed!")
    print("For comprehensive testing, run: python dynamic_constraint_rewards/test_all_dynamic_rewards.py")