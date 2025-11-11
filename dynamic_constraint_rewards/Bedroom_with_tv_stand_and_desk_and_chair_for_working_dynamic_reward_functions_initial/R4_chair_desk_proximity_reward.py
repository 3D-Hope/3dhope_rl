import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for chair-desk proximity.
    Returns higher reward when chair is close to desk (within 1.0m center-to-center).
    Uses smooth exponential decay: reward = exp(-distance^2 / (2 * sigma^2))
    where sigma = 0.5m, giving ~0.61 reward at 0.5m, ~0.14 at 1.0m.
    
    Input:
        - parsed_scenes: dict with scene tensors
        - idx_to_labels: dictionary mapping class indices to class labels
        - room_type: string
        - floor_polygons: list of floor polygon vertices
        - **kwargs: additional keyword arguments
    
    Output:
        reward: torch.Tensor of shape (B,)
    '''
    utility_functions = get_all_utility_functions()
    
    positions = parsed_scenes['positions']  # (B, N, 3)
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N = positions.shape[:2]
    device = parsed_scenes['device']
    
    # Get class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx['chair']
    desk_idx = labels_to_idx['desk']
    
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find chairs and desks in this scene
        chair_mask = (one_hot[b, :, chair_idx] == 1) & (~is_empty[b])
        desk_mask = (one_hot[b, :, desk_idx] == 1) & (~is_empty[b])
        
        chair_indices = torch.where(chair_mask)[0]
        desk_indices = torch.where(desk_mask)[0]
        
        if len(chair_indices) == 0 or len(desk_indices) == 0:
            # No chair or desk present, reward is 0
            rewards[b] = 0.0
            continue
        
        # Find minimum distance between any chair-desk pair
        min_distance = float('inf')
        for chair_idx_val in chair_indices:
            chair_pos = positions[b, chair_idx_val]  # (3,)
            for desk_idx_val in desk_indices:
                desk_pos = positions[b, desk_idx_val]  # (3,)
                # Compute XZ distance (ignore Y)
                xz_distance = torch.sqrt((chair_pos[0] - desk_pos[0])**2 + (chair_pos[2] - desk_pos[2])**2)
                min_distance = min(min_distance, xz_distance.item())
        
        # Compute reward using Gaussian decay
        sigma = 0.5  # Standard deviation in meters
        if min_distance == float('inf'):
            reward_val = 0.0
        else:
            # Exponential decay: high reward for close proximity
            reward_val = torch.exp(torch.tensor(-min_distance**2 / (2 * sigma**2), device=device))
            reward_val = reward_val.item()
        
        rewards[b] = reward_val
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the chair-desk proximity reward function.
    '''
    utility_functions = get_all_utility_functions()
    
    # Scene 1: Chair very close to desk (0.3m apart)
    num_objects_1 = 2
    class_label_indices_1 = [4, 7]  # chair, desk
    translations_1 = [(0, 0.4, 0), (0.3, 0.4, 0)]
    sizes_1 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1
    )
    
    # Scene 2: Chair far from desk (3m apart)
    num_objects_2 = 2
    class_label_indices_2 = [4, 7]  # chair, desk
    translations_2 = [(0, 0.4, 0), (3, 0.4, 0)]
    sizes_2 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2
    )
    
    # Scene 3: No desk (only chair)
    num_objects_3 = 1
    class_label_indices_3 = [4]  # chair
    translations_3 = [(0, 0.4, 0)]
    sizes_3 = [(0.3, 0.4, 0.3)]
    orientations_3 = [(1, 0)]
    scene_3 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3
    )
    
    # Stack scenes
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']
    
    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    # assert rewards.shape[0] == 3
    
    # Test # assertions
    print(f"Scene 1 (chair 0.3m from desk): {rewards[0].item():.4f}, expected: ~0.90-0.98")
    print(f"Scene 2 (chair 3m from desk): {rewards[1].item():.4f}, expected: ~0.00-0.01")
    print(f"Scene 3 (no desk): {rewards[2].item():.4f}, expected: 0.0")
    
    # assert rewards[0].item() > 0.85, f"Scene 1 should have high reward (>0.85), got {rewards[0].item()}"
    # assert rewards[1].item() < 0.05, f"Scene 2 should have low reward (<0.05), got {rewards[1].item()}"
    # assert rewards[2].item() == 0.0, f"Scene 3 should have reward 0.0, got {rewards[2].item()}"
    print("All tests passed!")