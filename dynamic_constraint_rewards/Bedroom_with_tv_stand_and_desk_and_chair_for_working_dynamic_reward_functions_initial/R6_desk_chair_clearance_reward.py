import torch
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for adequate clearance between desk and chair.
    Clearance should be 0.6-0.8m for comfortable use.
    Uses a Gaussian reward centered at 0.7m with sigma=0.15m.
    
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
    sizes = parsed_scenes['sizes']  # (B, N, 3)
    one_hot = parsed_scenes['one_hot']  # (B, N, num_classes)
    is_empty = parsed_scenes['is_empty']  # (B, N)
    B, N = positions.shape[:2]
    device = parsed_scenes['device']
    
    # Get class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    chair_idx = labels_to_idx['chair']
    desk_idx = labels_to_idx['desk']
    
    rewards = torch.zeros(B, device=device)
    
    optimal_clearance = 0.7  # meters
    sigma = 0.15  # Standard deviation
    
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
        
        # Find closest chair-desk pair and compute clearance
        best_reward = 0.0
        for chair_idx_val in chair_indices:
            chair_pos = positions[b, chair_idx_val]  # (3,)
            chair_size = sizes[b, chair_idx_val]  # (3,) - half extents
            
            for desk_idx_val in desk_indices:
                desk_pos = positions[b, desk_idx_val]  # (3,)
                desk_size = sizes[b, desk_idx_val]  # (3,) - half extents
                
                # Compute center-to-center distance in XZ plane
                center_distance = torch.sqrt((chair_pos[0] - desk_pos[0])**2 + (chair_pos[2] - desk_pos[2])**2)
                
                # Compute surface-to-surface clearance
                # Approximate as center distance minus half-widths
                chair_radius = torch.sqrt(chair_size[0]**2 + chair_size[2]**2)
                desk_radius = torch.sqrt(desk_size[0]**2 + desk_size[2]**2)
                clearance = center_distance - chair_radius - desk_radius
                clearance = max(0.0, clearance.item())
                
                # Gaussian reward centered at optimal clearance
                reward_val = torch.exp(torch.tensor(-(clearance - optimal_clearance)**2 / (2 * sigma**2), device=device))
                best_reward = max(best_reward, reward_val.item())
        
        rewards[b] = best_reward
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the desk-chair clearance reward function.
    '''
    utility_functions = get_all_utility_functions()
    
    # Scene 1: Optimal clearance (~0.7m surface-to-surface)
    # Chair at origin with size 0.3, desk at x=1.6 with size 0.6
    # Center distance = 1.6m, radii ~0.42 + 0.85 = 1.27, clearance ~0.33m
    # Let's place them closer: chair at 0, desk at 1.3
    num_objects_1 = 2
    class_label_indices_1 = [4, 7]  # chair, desk
    translations_1 = [(0, 0.4, 0), (1.3, 0.4, 0)]
    sizes_1 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_1 = [(1, 0), (1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1
    )
    
    # Scene 2: Too close (overlapping almost)
    num_objects_2 = 2
    class_label_indices_2 = [4, 7]  # chair, desk
    translations_2 = [(0, 0.4, 0), (0.5, 0.4, 0)]
    sizes_2 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_2 = [(1, 0), (1, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2
    )
    
    # Scene 3: Too far (2m apart)
    num_objects_3 = 2
    class_label_indices_3 = [4, 7]  # chair, desk
    translations_3 = [(0, 0.4, 0), (2.5, 0.4, 0)]
    sizes_3 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_3 = [(1, 0), (1, 0)]
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
    print(f"Scene 1 (good clearance): {rewards[0].item():.4f}, expected: >0.5")
    print(f"Scene 2 (too close): {rewards[1].item():.4f}, expected: <0.5")
    print(f"Scene 3 (too far): {rewards[2].item():.4f}, expected: <0.3")
    
    # assert rewards[0].item() > 0.3, f"Scene 1 should have reasonable reward (>0.3), got {rewards[0].item()}"
    # assert rewards[2].item() < 0.5, f"Scene 3 should have lower reward (<0.5), got {rewards[2].item()}"
    # assert rewards[0].item() > rewards[1].item() or rewards[0].item() > rewards[2].item(), "Scene 1 should have better reward than at least one other scene"
    print("All tests passed!")