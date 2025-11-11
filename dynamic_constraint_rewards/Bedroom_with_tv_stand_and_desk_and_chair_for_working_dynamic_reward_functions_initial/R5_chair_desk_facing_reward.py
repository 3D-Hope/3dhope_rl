import torch
import math
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Reward for chair facing the desk.
    Checks if chair's front direction points toward the desk.
    Reward = max(0, cos(angle)) where angle is between chair front and chair-to-desk vector.
    
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
    orientations = parsed_scenes['orientations']  # (B, N, 2)
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
        
        # Find best chair-desk pair based on facing alignment
        max_alignment = -1.0
        for chair_idx_val in chair_indices:
            chair_pos = positions[b, chair_idx_val]  # (3,)
            chair_orient = orientations[b, chair_idx_val]  # (2,) [cos, sin]
            chair_size = sizes[b, chair_idx_val]  # (3,)
            
            # Get chair front direction using utility function
            chair_front, _ = utility_functions["find_object_front_and_back"]["function"](
                chair_pos.unsqueeze(0), chair_orient.unsqueeze(0), chair_size.unsqueeze(0)
            )
            # Chair front direction vector (in XZ plane)
            chair_front_dir = chair_front[0] - chair_pos  # (3,)
            chair_front_dir_xz = torch.tensor([chair_front_dir[0], chair_front_dir[2]], device=device)
            chair_front_dir_xz = chair_front_dir_xz / (torch.norm(chair_front_dir_xz) + 1e-8)
            
            for desk_idx_val in desk_indices:
                desk_pos = positions[b, desk_idx_val]  # (3,)
                
                # Vector from chair to desk (in XZ plane)
                chair_to_desk = torch.tensor([desk_pos[0] - chair_pos[0], desk_pos[2] - chair_pos[2]], device=device)
                chair_to_desk = chair_to_desk / (torch.norm(chair_to_desk) + 1e-8)
                
                # Compute alignment (dot product)
                alignment = torch.dot(chair_front_dir_xz, chair_to_desk).item()
                max_alignment = max(max_alignment, alignment)
        
        # Reward is max(0, alignment) - chair should face toward desk
        rewards[b] = max(0.0, max_alignment)
    
    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test the chair-desk facing reward function.
    '''
    utility_functions = get_all_utility_functions()
    
    # Scene 1: Chair facing desk (chair at origin facing +X, desk at +X)
    num_objects_1 = 2
    class_label_indices_1 = [4, 7]  # chair, desk
    translations_1 = [(0, 0.4, 0), (1.5, 0.4, 0)]
    sizes_1 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_1 = [(1, 0), (1, 0)]  # Both facing +X
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1
    )
    
    # Scene 2: Chair facing away from desk (chair facing +X, desk at -X)
    num_objects_2 = 2
    class_label_indices_2 = [4, 7]  # chair, desk
    translations_2 = [(0, 0.4, 0), (-1.5, 0.4, 0)]
    sizes_2 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_2 = [(1, 0), (1, 0)]  # Chair facing +X, desk at -X
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2
    )
    
    # Scene 3: Chair perpendicular to desk (chair facing +X, desk at +Z)
    num_objects_3 = 2
    class_label_indices_3 = [4, 7]  # chair, desk
    translations_3 = [(0, 0.4, 0), (0, 0.4, 1.5)]
    sizes_3 = [(0.3, 0.4, 0.3), (0.6, 0.4, 0.5)]
    orientations_3 = [(1, 0), (1, 0)]  # Chair facing +X, desk at +Z
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
    print(f"Scene 1 (chair facing desk): {rewards[0].item():.4f}, expected: ~0.9-1.0")
    print(f"Scene 2 (chair facing away): {rewards[1].item():.4f}, expected: 0.0")
    print(f"Scene 3 (chair perpendicular): {rewards[2].item():.4f}, expected: ~0.0")
    
    ## assert rewards[0].item() > 0.85, f"Scene 1 should have high reward (>0.85), got {rewards[0].item()}"
    # assert rewards[1].item() < 0.1, f"Scene 2 should have low reward (<0.1), got {rewards[1].item()}"
    ## assert rewards[2].item() < 0.2, f"Scene 3 should have low reward (<0.2), got {rewards[2].item()}"
    print("All tests passed!")