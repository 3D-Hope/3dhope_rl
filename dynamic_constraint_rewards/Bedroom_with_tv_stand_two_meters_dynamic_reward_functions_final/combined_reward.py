import torch

from dynamic_constraint_rewards.utilities import get_all_utility_functions

def get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Reward for TV stand and bed being present, facing each other, and placed within 2 meters.

    Reward structure:
    - Bed and TV stand both present, facing each other, and within 2 meters: 10.0
    - Bed and TV stand both present, NOT properly facing or NOT within 2 meters: 2.0
    - Only bed or TV stand present: 0.0
    - Neither present: -2.0
    - Overplacement penalty: -2.0 per extra bed or TV stand beyond the first

    All checks are per batch entry.
    """
    utility_functions = get_all_utility_functions()
    
    overplacement_penalty = 2.0  # Penalty per extra object

    one_hot = parsed_scenes["one_hot"]  # (B, N, num_classes)
    positions = parsed_scenes["positions"]  # (B, N, 3)
    orientations = parsed_scenes["orientations"]  # (B, N, 2)
    idx_to_label = idx_to_labels

    B, N, num_classes = one_hot.shape[0], one_hot.shape[1], one_hot.shape[2]
    device = one_hot.device

    # Find indices for bed and tv_stand in the class mapping
    bed_label_names = ["bed", "single_bed", "double_bed"]  # support generic and specific
    tv_stand_label_names = ["tv_stand"]

    label_to_idx = {v: k for k, v in idx_to_label.items()}

    # Build mask per scene for bed and tv_stand
    bed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for label in bed_label_names:
        if label in label_to_idx:
            bed_mask |= (parsed_scenes["object_indices"] == label_to_idx[label])
    tv_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for label in tv_stand_label_names:
        if label in label_to_idx:
            tv_mask |= (parsed_scenes["object_indices"] == label_to_idx[label])
    has_bed = bed_mask.any(dim=1)        # (B,)
    has_tv = tv_mask.any(dim=1)          # (B,)

    both_present = has_bed & has_tv
    
    # Count number of beds and TV stands per scene
    num_beds = bed_mask.sum(dim=1)  # (B,)
    num_tvs = tv_mask.sum(dim=1)    # (B,)
    
    # Calculate overplacement penalty (expected: 1 bed, 1 TV stand)
    bed_overplacement = torch.clamp(num_beds - 1, min=0)  # Extra beds
    tv_overplacement = torch.clamp(num_tvs - 1, min=0)    # Extra TVs
    total_overplacement = bed_overplacement + tv_overplacement
    overplacement_penalty_value = -total_overplacement * overplacement_penalty

    # Default reward is -2
    rewards = torch.full((B,), -2.0, device=device)

    # Only one object present: reward 0
    rewards[(has_bed ^ has_tv)] = 0.0  # XOR: exactly one present

    # Check placement/relationship for scenes where both are present
    for b in range(B):
        if both_present[b]:
            bed_inds = torch.where(bed_mask[b])[0]
            tv_inds = torch.where(tv_mask[b])[0]
            # If there are multiple, check all possible pairs; reward based on "best" relation
            best_pair_reward = 2.0  # fallback: both present, but not facing or not close enough

            for bi in bed_inds:
                for tvi in tv_inds:
                    pos_bed = positions[b, bi]    # (3,)
                    pos_tv = positions[b, tvi]    # (3,)
                    orient_bed = orientations[b, bi]  # (2,)
                    orient_tv = orientations[b, tvi]  # (2,)

                    # Vector from bed to tv
                    vec_bed_to_tv = (pos_tv - pos_bed)
                    vec_bed_to_tv_xy = vec_bed_to_tv[[0,2]] # [Ashok] var name is wrong but logic is correct, y should be z
                    dist = torch.norm(vec_bed_to_tv_xy)

                    # Unit forward vectors (assume [cos, sin] theta defines +x axis; rotate if needed)
                    bed_forward = torch.tensor([orient_bed[0], orient_bed[1]], device=device)
                    tv_forward = torch.tensor([orient_tv[0], orient_tv[1]], device=device)

                    # Normalize
                    if bed_forward.norm() > 0:
                        bed_forward = bed_forward / bed_forward.norm()
                    if tv_forward.norm() > 0:
                        tv_forward = tv_forward / tv_forward.norm()

                    dir_bed_to_tv = vec_bed_to_tv_xy / (vec_bed_to_tv_xy.norm() + 1e-8)
                    dir_tv_to_bed = -dir_bed_to_tv

                    # Bed faces toward tv (dot > 0.9) and tv faces toward bed (dot > 0.9)
                    # You may adjust tolerance if you want some slop in "facing"
                    facing_ok = (torch.dot(bed_forward, dir_bed_to_tv) > 0.9) and \
                                (torch.dot(tv_forward, dir_tv_to_bed) > 0.9)
                    close_enough = (dist <= 2.0)

                    if facing_ok and close_enough:
                        best_pair_reward = 10.0
                        break  # no need to check more
            rewards[b] = best_pair_reward
    
    # Apply overplacement penalty to all rewards
    rewards = rewards + overplacement_penalty_value

    return rewards

def test_reward(idx_to_labels, room_type, floor_polygons, **kwargs):
    """
    Test reward for TV stand and bed presence, facing, and proximity.
    """
    utility_functions = get_all_utility_functions()

    # Helper: find the class index for tv_stand and bed
    label_to_idx = {v: k for k, v in idx_to_labels.items()}
    bed_idx = label_to_idx.get("bed", 8)
    tv_idx = label_to_idx.get("tv_stand", 19)

    # Scene 1: bed and tv_stand present, facing each other, within 2 meters (should be 10.0)
    num_objects_1 = 2
    class_label_indices_1 = [tv_idx, bed_idx]
    translations_1 = [(0, 0, 0), (2, 0, 0)]  # 2 meters apart in x
    sizes_1 = [(0.6, 0.5, 0.3), (2.0, 0.5, 1.0)]
    # tv_stand faces +x, bed faces -x
    orientations_1 = [(1, 0), (-1, 0)]
    scene_1 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_1, sizes_1, orientations_1
    )

    # Scene 2: bed and tv_stand present, wrong orientation (should be 2.0)
    orientations_2 = [(1, 0), (1, 0)]  # both facing +x
    translations_2 = [(0, 0, 0), (2, 0, 0)]
    scene_2 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_2, sizes_1, orientations_2
    )

    # Scene 3: bed and tv_stand, facing but far (> 2 meters, should be 2.0)
    translations_3 = [(0, 0, 0), (4, 0, 0)]
    orientations_3 = [(1, 0), (-1, 0)]
    scene_3 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_1, class_label_indices_1, translations_3, sizes_1, orientations_1
    )

    # Scene 4: only bed (should be 0.0)
    num_objects_4 = 1
    class_label_indices_4 = [bed_idx]
    translations_4 = [(0, 0, 0)]
    sizes_4 = [(2.0, 0.5, 1.0)]
    orientations_4 = [(1, 0)]
    scene_4 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_4, class_label_indices_4, translations_4, sizes_4, orientations_4
    )

    # Scene 5: neither bed nor tv_stand (should be -2.0)
    num_objects_5 = 1
    class_label_indices_5 = [12]  # nightstand
    translations_5 = [(3, 0, 3)]
    sizes_5 = [(0.5, 0.3, 0.5)]
    orientations_5 = [(1, 0)]
    scene_5 = utility_functions["create_scene_for_testing"]["function"](
        room_type, num_objects_5, class_label_indices_5, translations_5, sizes_5, orientations_5
    )

    # stack all scenes
    tensor_keys = [k for k in scene_1 if isinstance(scene_1[k], torch.Tensor)]
    parsed_scenes = {
        k: torch.cat([scene_1[k], scene_2[k], scene_3[k], scene_4[k], scene_5[k]], dim=0)
        for k in tensor_keys
    }
    parsed_scenes['room_type'] = room_type
    parsed_scenes['device'] = scene_1['device']

    rewards = get_reward(parsed_scenes, idx_to_labels, room_type, floor_polygons, **kwargs)
    print("Rewards:", rewards)
    print("Expected: [10.0, 2.0, 2.0, 0.0, -2.0]")
    expected = [10.0, 2.0, 2.0, 0.0, -2.0]
    assert list(rewards.shape) == [5]
    for i in range(5):
        assert torch.isclose(rewards[i], torch.tensor(expected[i], dtype=rewards.dtype, device=rewards.device)), f"Scene {i+1} expected {expected[i]}, got {rewards[i].item()}"

    print("All tests passed!")


if __name__ == "__main__":
    room_type = "Bedroom"
    idx_to_labels = {
        0: "empty",
        1: "arm_chair",
        2: "bookshelf",
        3: "cabinet",
        4: "chair",
        5: "desk",
        6: "dining_table",
        7: "double_bed",
        8: "bed",
        9: "nightstand",
        10: "sofa",
        11: "table",
        12: "tv_stand",
        13: "wardrobe",
    }
    test_reward(idx_to_labels, room_type, floor_polygons)