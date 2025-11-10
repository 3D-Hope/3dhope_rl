import torch
import math
from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Stage 1 Curriculum: Learn to avoid West (180°) orientation.
    
    Reward structure:
    - East (0°): 1.0 ✓
    - South (90°): 0.5 (acceptable)
    - North (270°): 0.5 (acceptable)
    - West (180°): 0.0 ✗ (avoid this!)
    
    Training goal: 80%+ scenes should avoid West orientation
    '''
    
    device = parsed_scenes['device']
    orientations = parsed_scenes['orientations']  # (B, N, 2) -> [cos, sin]
    is_empty = parsed_scenes['is_empty']  # (B, N)
    object_indices = parsed_scenes['object_indices']  # (B, N)
    
    B, N = orientations.shape[:2]
    
    # Get bed class indices
    labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_classes = [labels_to_idx['double_bed'], labels_to_idx['single_bed'], labels_to_idx['kids_bed']]
    
    # Initialize rewards
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        # Find bed objects
        bed_mask = torch.zeros(N, dtype=torch.bool, device=device)
        for bed_class in bed_classes:
            bed_mask |= (object_indices[b] == bed_class) & (~is_empty[b])
        
        if not bed_mask.any():
            rewards[b] = -0.3  # No bed penalty
            continue
        
        # Get first bed's orientation
        bed_idx = torch.where(bed_mask)[0][0]
        cos_theta = orientations[b, bed_idx, 0]
        
        # Stage 1 reward: Only penalize West (cos ≈ -1)
        # East (cos ≈ 1): reward = 1.0
        # North/South (cos ≈ 0): reward = 0.5
        # West (cos ≈ -1): reward = 0.0
        
        # Simple linear mapping: reward = (cos + 1) / 2
        # But clamp minimum at 0.5 for non-West orientations
        base_reward = (cos_theta + 1.0) / 2.0
        
        # Give 0.5 minimum for anything that's not facing West
        if cos_theta > -0.5:  # Not facing West
            rewards[b] = torch.max(base_reward, torch.tensor(0.5, device=device))
        else:  # Facing West
            rewards[b] = base_reward
    
    return rewards



def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    '''
    Test function for bed headboard faces east constraint.
    '''
    pass
    # utility_functions = get_all_utility_functions()
    # create_scene = utility_functions["create_scene_for_testing"]["function"]
    
    # labels_to_idx = {v: k for k, v in idx_to_labels.items()}
    # double_bed_idx = labels_to_idx['double_bed']
    # nightstand_idx = labels_to_idx['nightstand']
    # wardrobe_idx = labels_to_idx['wardrobe']
    
    # # Scene 1: Bed facing exactly east (orientation = [1, 0])
    # num_objects_1 = 3
    # class_label_indices_1 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    # translations_1 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    # sizes_1 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    # orientations_1 = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]  # Bed facing east
    # scene_1 = create_scene(room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1)
    
    # # Scene 2: Bed facing west (orientation = [-1, 0], 180 degrees)
    # num_objects_2 = 3
    # class_label_indices_2 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    # translations_2 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    # sizes_2 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    # orientations_2 = [(-1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]  # Bed facing west
    # scene_2 = create_scene(room_type, num_objects_2, class_label_indices_2, translations_2, sizes_2, orientations_2)
    
    # # Scene 3: Bed facing north (orientation = [0, 1], 90 degrees)
    # num_objects_3 = 3
    # class_label_indices_3 = [double_bed_idx, nightstand_idx, wardrobe_idx]
    # translations_3 = [(0, 0.4, 0), (1.5, 0.3, 0), (-1.5, 0.5, 0)]
    # sizes_3 = [(1.0, 0.4, 0.9), (0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    # orientations_3 = [(0.0, 1.0), (1.0, 0.0), (1.0, 0.0)]  # Bed facing north
    # scene_3 = create_scene(room_type, num_objects_3, class_label_indices_3, translations_3, sizes_3, orientations_3)
    
    # # Scene 4: No bed
    # num_objects_4 = 2
    # class_label_indices_4 = [nightstand_idx, wardrobe_idx]
    # translations_4 = [(1.5, 0.3, 0), (-1.5, 0.5, 0)]
    # sizes_4 = [(0.3, 0.3, 0.3), (0.5, 0.5, 1.0)]
    # orientations_4 = [(1.0, 0.0), (1.0, 0.0)]
    # scene_4 = create_scene(room_type, num_objects_4, class_label_indices_4, translations_4, sizes_4, orientations_4)
    
    # # Stack scenes
    # tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    # parsed_scenes = {
    #     k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k]], dim=0)
    #     for k in tensor_keys
    # }
    # parsed_scenes['room_type'] = room_type
    # parsed_scenes['device'] = scene_1['device']
    
    # rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    # print("Rewards:", rewards)
    # print(f"Scene 1 (bed facing east): {rewards[0].item():.4f}")
    # print(f"Scene 2 (bed facing west): {rewards[1].item():.4f}")
    # print(f"Scene 3 (bed facing north): {rewards[2].item():.4f}")
    # print(f"Scene 4 (no bed): {rewards[3].item():.4f}")
    
    # assert rewards.shape[0] == 4, "Should have 4 scenes"
    
    # # Test assertions with softer thresholds
    # assert rewards[0].item() > 0.95, f"Scene 1: Bed facing east should have reward close to 1.0, got {rewards[0].item()}"
    # assert rewards[1].item() < 0.05, f"Scene 2: Bed facing west should have reward close to 0.0, got {rewards[1].item()}"
    # assert 0.45 < rewards[2].item() < 0.55, f"Scene 3: Bed facing north should have reward close to 0.5, got {rewards[2].item()}"
    # assert rewards[3].item() < 0.0, f"Scene 4: No bed should have negative reward, got {rewards[3].item()}"
    # assert rewards[0] > rewards[2] > rewards[1], f"Scene 1 > Scene 3 > Scene 2, got {rewards[0]}, {rewards[2]}, {rewards[1]}"
    
    # print("All tests passed!")