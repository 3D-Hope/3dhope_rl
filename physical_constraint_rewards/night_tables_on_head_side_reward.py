"""
Night Tables on Head Side Reward - Ensures nightstands are placed correctly relative to beds.

Bed orientation convention (same as chair_accessibility_reward.py):
- Forward direction = direction you'd look when sitting behind headboard
- For θ=0: Forward is +Z (foot of bed), so head is at -Z (behind headboard)
- Nightstands should be on the HEAD side (opposite of forward direction)

Reward cases:
1. Single nightstand on head side: +0.5
2. Two nightstands on both sides of head: +1.0 (bonus for symmetry)
3. Nightstand on foot side (wrong): -0.5

FIXED ISSUES:
- Now properly checks lateral distance (nightstands can be farther to the sides)
- Ensures nightstands are actually on LEFT or RIGHT, not centered
- Fixed symmetric bonus to require nightstands on DIFFERENT sides
"""

import torch
import torch.nn.functional as F


def compute_facing_direction(theta):
    """
    Compute forward direction vector from orientation angle.
    
    For beds: forward = foot direction (where you'd look when sitting behind headboard)
    Head direction = -forward
    
    Args:
        theta: (B, N) tensor of orientation angles in radians
    
    Returns:
        forward: (B, N, 2) tensor of unit vectors in XZ plane [dx, dz]
    """
    dx = -torch.sin(theta)  # X component
    dz = torch.cos(theta)   # Z component
    
    forward = torch.stack([dx, dz], dim=-1)  # (B, N, 2)
    return forward


def get_bed_head_foot_sides(bed_positions, bed_sizes, bed_orientations):
    """
    Determine the head and foot side regions of beds.
    
    For a bed at position (x, z) with orientation θ:
    - Forward (foot) direction: [dx, dz] = [-sin(θ), cos(θ)]
    - Head direction: -[dx, dz]
    - Left/right perpendicular: [-dz, dx] (90° rotation)
    
    Head side region: bed_center - forward * (bed_length/2)
    Foot side region: bed_center + forward * (bed_length/2)
    
    Args:
        bed_positions: (N_beds, 3) positions [x, y, z]
        bed_sizes: (N_beds, 3) half-extents [sx, sy, sz]
        bed_orientations: (N_beds,) angles in radians
    
    Returns:
        dict with:
            - head_centers: (N_beds, 2) XZ positions of head side centers
            - foot_centers: (N_beds, 2) XZ positions of foot side centers
            - left_dirs: (N_beds, 2) left perpendicular directions
            - right_dirs: (N_beds, 2) right perpendicular directions
            - bed_widths: (N_beds,) half-width of beds
            - bed_half_lengths: (N_beds,) half-length of beds
    """
    N_beds = bed_positions.shape[0]
    device = bed_positions.device
    
    if N_beds == 0:
        return {
            'head_centers': torch.zeros(0, 2, device=device),
            'foot_centers': torch.zeros(0, 2, device=device),
            'left_dirs': torch.zeros(0, 2, device=device),
            'right_dirs': torch.zeros(0, 2, device=device),
            'bed_widths': torch.zeros(0, device=device),
            'bed_half_lengths': torch.zeros(0, device=device),
            'forward_dirs': torch.zeros(0, 2, device=device),
            'head_dirs': torch.zeros(0, 2, device=device),
        }
    
    # Get forward directions (foot direction)
    forward_dirs = compute_facing_direction(bed_orientations)  # (N_beds, 2)
    
    # Head direction is opposite of forward
    head_dirs = -forward_dirs
    
    # Perpendicular directions (left/right when looking at foot from head)
    # Left = rotate forward 90° CCW: [dx, dz] -> [-dz, dx]
    left_dirs = torch.stack([-forward_dirs[:, 1], forward_dirs[:, 0]], dim=-1)
    right_dirs = -left_dirs
    
    # Extract XZ positions
    bed_xz = bed_positions[:, [0, 2]]  # (N_beds, 2)
    
    # Bed dimensions in XZ plane
    # Assume bed length is along Z-axis before rotation (typically beds are longer than wide)
    # After rotation, the half-length along the forward direction
    # For simplicity, use max(sx, sz) as half-length
    bed_half_lengths = torch.max(bed_sizes[:, 0], bed_sizes[:, 2])  # (N_beds,)
    bed_widths = torch.min(bed_sizes[:, 0], bed_sizes[:, 2])  # (N_beds,)
    
    # Head and foot centers
    head_centers = bed_xz + head_dirs * bed_half_lengths.unsqueeze(-1)  # (N_beds, 2)
    foot_centers = bed_xz + forward_dirs * bed_half_lengths.unsqueeze(-1)  # (N_beds, 2)
    
    return {
        'head_centers': head_centers,
        'foot_centers': foot_centers,
        'left_dirs': left_dirs,
        'right_dirs': right_dirs,
        'bed_widths': bed_widths,
        'bed_half_lengths': bed_half_lengths,
        'forward_dirs': forward_dirs,
        'head_dirs': head_dirs,
    }


def check_nightstand_placement(nightstand_positions, bed_info, bed_positions, 
                               proximity_threshold=1.5, lateral_threshold=2.0, debug=False):
    """
    Check if nightstands are correctly placed on the head side of beds.
    
    A nightstand is correctly placed if:
    1. It's on the HEAD side (not foot side)
    2. It's within proximity_threshold along the head-foot axis
    3. It's on the left or right SIDE (not centered, with lateral distance check)
    4. It's within lateral_threshold distance to the sides
    
    Args:
        nightstand_positions: (N_nightstands, 3) positions
        bed_info: dict from get_bed_head_foot_sides()
        bed_positions: (N_beds, 3) original bed positions
        proximity_threshold: Maximum distance along head-foot axis (default 1.5m)
        lateral_threshold: Maximum lateral distance to sides (default 2.0m)
        debug: Print debug information
    
    Returns:
        dict with:
            - on_head_side: (N_nightstands, N_beds) boolean mask
            - on_foot_side: (N_nightstands, N_beds) boolean mask
            - on_left_side: (N_nightstands, N_beds) boolean mask
            - on_right_side: (N_nightstands, N_beds) boolean mask
            - distances: (N_nightstands, N_beds) distances to bed centers
    """
    N_nightstands = nightstand_positions.shape[0]
    N_beds = bed_positions.shape[0]
    device = nightstand_positions.device
    
    if N_nightstands == 0 or N_beds == 0:
        return {
            'on_head_side': torch.zeros(N_nightstands, N_beds, dtype=torch.bool, device=device),
            'on_foot_side': torch.zeros(N_nightstands, N_beds, dtype=torch.bool, device=device),
            'on_left_side': torch.zeros(N_nightstands, N_beds, dtype=torch.bool, device=device),
            'on_right_side': torch.zeros(N_nightstands, N_beds, dtype=torch.bool, device=device),
            'distances': torch.full((N_nightstands, N_beds), float('inf'), device=device),
        }
    
    # Extract XZ coordinates
    nightstand_xz = nightstand_positions[:, [0, 2]]  # (N_nightstands, 2)
    bed_xz = bed_positions[:, [0, 2]]  # (N_beds, 2)
    
    # Compute vectors from bed centers to nightstands
    # (N_nightstands, 1, 2) - (1, N_beds, 2) -> (N_nightstands, N_beds, 2)
    bed_to_nightstand = nightstand_xz.unsqueeze(1) - bed_xz.unsqueeze(0)
    
    # Total distances
    distances = torch.norm(bed_to_nightstand, dim=-1)  # (N_nightstands, N_beds)
    
    # Normalize vectors
    bed_to_nightstand_norm = F.normalize(bed_to_nightstand, dim=-1, eps=1e-8)
    
    # Expand bed directions for broadcasting
    head_dirs = bed_info['head_dirs'].unsqueeze(0)  # (1, N_beds, 2)
    forward_dirs = bed_info['forward_dirs'].unsqueeze(0)  # (1, N_beds, 2)
    left_dirs = bed_info['left_dirs'].unsqueeze(0)  # (1, N_beds, 2)
    right_dirs = bed_info['right_dirs'].unsqueeze(0)  # (1, N_beds, 2)
    bed_half_lengths = bed_info['bed_half_lengths'].unsqueeze(0)  # (1, N_beds)
    bed_widths = bed_info['bed_widths'].unsqueeze(0)  # (1, N_beds)
    
    # Dot products with each direction
    dot_head = (bed_to_nightstand_norm * head_dirs).sum(dim=-1)  # (N_nightstands, N_beds)
    dot_foot = (bed_to_nightstand_norm * forward_dirs).sum(dim=-1)
    dot_left = (bed_to_nightstand_norm * left_dirs).sum(dim=-1)
    dot_right = (bed_to_nightstand_norm * right_dirs).sum(dim=-1)
    
    # Calculate distances along bed axes
    # Distance along head-foot axis (forward direction)
    dist_along_head_foot = torch.abs((bed_to_nightstand * forward_dirs).sum(dim=-1))  # (N_nightstands, N_beds)
    
    # Distance along lateral axis (left-right direction)
    dist_lateral = torch.abs((bed_to_nightstand * left_dirs).sum(dim=-1))  # (N_nightstands, N_beds)
    
    if debug:
        for i in range(N_nightstands):
            for j in range(N_beds):
                print(f"      Nightstand {i} -> Bed {j}:")
                print(f"        Total dist={distances[i,j]:.2f}m")
                print(f"        Dist along head-foot={dist_along_head_foot[i,j]:.2f}m")
                print(f"        Dist lateral={dist_lateral[i,j]:.2f}m")
                print(f"        dot_head={dot_head[i,j]:.2f}, dot_foot={dot_foot[i,j]:.2f}")
                print(f"        dot_left={dot_left[i,j]:.2f}, dot_right={dot_right[i,j]:.2f}")
    
    # Nightstand is on head side if:
    # 1. Points toward head (dot_head > 0.3)
    # 2. Within proximity threshold along head-foot axis
    # 3. Has significant lateral distance (not centered)
    # 4. Within lateral threshold
    on_head_side = (
        (dot_head > 0.3) & 
        (dist_along_head_foot < proximity_threshold + bed_half_lengths) &
        (dist_lateral > bed_widths * 0.3) &  # Must be to the side, not center
        (dist_lateral < lateral_threshold)
    )
    
    # Nightstand is on foot side if pointing toward foot
    on_foot_side = (
        (dot_foot > 0.3) & 
        (dist_along_head_foot < proximity_threshold + bed_half_lengths) &
        (dist_lateral < lateral_threshold)
    )
    
    # Left/right side determination (only if on head side)
    on_left_side = (dot_left > 0.5) & on_head_side
    on_right_side = (dot_right > 0.5) & on_head_side
    
    return {
        'on_head_side': on_head_side,
        'on_foot_side': on_foot_side,
        'on_left_side': on_left_side,
        'on_right_side': on_right_side,
        'distances': distances,
        'dot_head': dot_head,
        'dot_left': dot_left,
        'dot_right': dot_right,
        'dist_along_head_foot': dist_along_head_foot,
        'dist_lateral': dist_lateral,
    }


def compute_night_tables_reward(parsed_scene, head_side_bonus=0.5, 
                                both_sides_bonus=1.0, wrong_side_penalty=-0.5,
                                debug=False, **kwargs):
    """
    Calculate reward for nightstand placement relative to beds.
    
    Rewards:
    - Nightstand on head side: +head_side_bonus (default +0.5)
    - Two nightstands on both sides of head: +both_sides_bonus (default +1.0)
    
    Penalties:
    - Nightstand on foot side: +wrong_side_penalty (default -0.5)
    
    Args:
        parsed_scene: Dict from parse_and_descale_scenes()
        head_side_bonus: Reward for nightstand on head side
        both_sides_bonus: Bonus for symmetric placement
        wrong_side_penalty: Penalty for nightstand on wrong side
        debug: Print debug information
        **kwargs: Additional arguments
    
    Returns:
        rewards: (B,) tensor with per-scene rewards
    """
    positions = parsed_scene['positions']  # (B, N, 3)
    sizes = parsed_scene['sizes']  # (B, N, 3) - HALF-EXTENTS
    orientations_cos_sin = parsed_scene['orientations']  # (B, N, 2)
    object_indices = parsed_scene['object_indices']  # (B, N)
    is_empty = parsed_scene['is_empty']  # (B, N)
    
    B, N = positions.shape[:2]
    device = positions.device
    
    # Convert cos/sin to radians
    cos_theta = orientations_cos_sin[:, :, 0]
    sin_theta = orientations_cos_sin[:, :, 1]
    orientations = torch.atan2(sin_theta, cos_theta)  # (B, N)
    
    # Identify beds (single_bed=15, double_bed=8, kids_bed=11)
    bed_classes = [8, 11, 15]
    is_bed = torch.zeros(B, N, dtype=torch.bool, device=device)
    for cls_id in bed_classes:
        is_bed |= (object_indices == cls_id)
    is_bed = is_bed & (~is_empty)
    
    # Identify nightstands (nightstand=12)
    nightstand_class = 12
    is_nightstand = (object_indices == nightstand_class) & (~is_empty)
    
    # Compute rewards per scene
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        bed_mask = is_bed[b]
        nightstand_mask = is_nightstand[b]
        
        n_beds = bed_mask.sum().item()
        n_nightstands = nightstand_mask.sum().item()
        
        if debug:
            print(f"\nScene {b}: {n_beds} beds, {n_nightstands} nightstands")
        
        if n_beds == 0 or n_nightstands == 0:
            # No beds or no nightstands - neutral reward
            continue
        
        # Extract bed and nightstand data
        bed_positions_b = positions[b, bed_mask]  # (n_beds, 3)
        bed_sizes_b = sizes[b, bed_mask]  # (n_beds, 3)
        bed_orientations_b = orientations[b, bed_mask]  # (n_beds,)
        
        nightstand_positions_b = positions[b, nightstand_mask]  # (n_nightstands, 3)
        
        # Get bed geometry
        bed_info = get_bed_head_foot_sides(bed_positions_b, bed_sizes_b, bed_orientations_b)
        
        if debug:
            for i in range(n_beds):
                print(f"  Bed {i}: pos=({bed_positions_b[i,0]:.2f}, {bed_positions_b[i,2]:.2f}), "
                      f"size=({bed_sizes_b[i,0]:.2f}, {bed_sizes_b[i,2]:.2f}), "
                      f"θ={bed_orientations_b[i]*180/3.14159:.1f}°")
                print(f"    Head center: ({bed_info['head_centers'][i,0]:.2f}, {bed_info['head_centers'][i,1]:.2f})")
                print(f"    Foot center: ({bed_info['foot_centers'][i,0]:.2f}, {bed_info['foot_centers'][i,1]:.2f})")
                print(f"    Forward dir: ({bed_info['forward_dirs'][i,0]:.2f}, {bed_info['forward_dirs'][i,1]:.2f})")
        
        # Check nightstand placements
        placement = check_nightstand_placement(nightstand_positions_b, bed_info, bed_positions_b, debug=debug)
        
        # Count correct and incorrect placements
        # A nightstand is "correct" if it's on the head side of at least one bed
        correct_placements = placement['on_head_side'].any(dim=1)  # (n_nightstands,)
        wrong_placements = placement['on_foot_side'].any(dim=1) & (~correct_placements)
        
        n_correct = correct_placements.sum().float()
        n_wrong = wrong_placements.sum().float()
        
        # Check for symmetric placement (both left and right)
        # For each bed, check if it has nightstands on both left AND right sides
        # FIXED: Ensure nightstands are on DIFFERENT sides, not both on same side
        n_symmetric = 0
        for bed_idx in range(n_beds):
            # Check which nightstands are on left/right for this bed
            ns_on_left = placement['on_left_side'][:, bed_idx] & placement['on_head_side'][:, bed_idx]
            ns_on_right = placement['on_right_side'][:, bed_idx] & placement['on_head_side'][:, bed_idx]
            
            has_left = ns_on_left.any()
            has_right = ns_on_right.any()
            
            if has_left and has_right:
                n_symmetric += 1
        
        if debug:
            print(f"  Correct placements: {n_correct.item()}")
            print(f"  Wrong placements: {n_wrong.item()}")
            print(f"  Symmetric beds: {n_symmetric}")
            for i, night_pos in enumerate(nightstand_positions_b):
                for j, bed_pos in enumerate(bed_positions_b):
                    if placement['on_head_side'][i, j]:
                        side = "LEFT" if placement['on_left_side'][i, j] else "RIGHT" if placement['on_right_side'][i, j] else "CENTER"
                        print(f"    Nightstand {i} -> Bed {j}: HEAD side, {side}")
                        print(f"      dist_total={placement['distances'][i,j]:.2f}m, "
                              f"dist_head_foot={placement['dist_along_head_foot'][i,j]:.2f}m, "
                              f"dist_lateral={placement['dist_lateral'][i,j]:.2f}m")
                    elif placement['on_foot_side'][i, j]:
                        print(f"    Nightstand {i} -> Bed {j}: FOOT side (wrong!)")
        
        # Compute reward
        reward = n_correct * head_side_bonus + n_wrong * wrong_side_penalty + n_symmetric * both_sides_bonus
        
        # Normalize by number of nightstands
        if n_nightstands > 0:
            rewards[b] = reward / n_nightstands
    
    return rewards


def test_night_tables_reward():
    """Test cases for nightstand placement reward."""
    print("\n" + "="*70)
    print("Testing Night Tables Reward (FIXED VERSION)")
    print("="*70)
    
    device = 'cpu'
    num_classes = 22
    num_objects = 12
    
    from physical_constraint_rewards.commons import parse_and_descale_scenes
    
    # Room bounds for normalization
    x_min, x_max = -2.7625005, 2.7784417
    z_min, z_max = -2.75275, 2.8185427
    y_min, y_max = 0.045, 3.6248395
    
    def normalize_coord(val, min_val, max_val):
        return 2 * (val - min_val) / (max_val - min_val) - 1
    
    def create_scene(bed_config, nightstand_configs):
        """Create scene with bed and nightstands."""
        scene = torch.zeros(num_objects, 30, device=device)
        idx = 0
        
        # Add bed
        x, z, theta_deg = bed_config
        theta_rad = torch.tensor(theta_deg * 3.14159 / 180.0, device=device)
        scene[idx, 8] = 1.0  # Double bed (class=8)
        x_norm = normalize_coord(x, x_min, x_max)
        z_norm = normalize_coord(z, z_min, z_max)
        y_norm = normalize_coord(0.5, y_min, y_max)
        scene[idx, 22:25] = torch.tensor([x_norm, y_norm, z_norm], device=device)
        # Bed size: ~2m long, ~1.5m wide (half-extents: 1.0, 0.3, 0.75)
        size_min = torch.tensor([0.03998289, 0.02000002, 0.012772], device=device)
        size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)
        sx_norm = normalize_coord(0.75, size_min[0], size_max[0])
        sy_norm = normalize_coord(0.3, size_min[1], size_max[1])
        sz_norm = normalize_coord(1.0, size_min[2], size_max[2])
        scene[idx, 25:28] = torch.tensor([sx_norm, sy_norm, sz_norm], device=device)
        scene[idx, 28] = torch.cos(theta_rad)
        scene[idx, 29] = torch.sin(theta_rad)
        print(f"  Bed at ({x:.2f}, {z:.2f}), θ={theta_deg}° (head at -forward)")
        idx += 1
        
        # Add nightstands
        for ns_x, ns_z in nightstand_configs:
            scene[idx, 12] = 1.0  # Nightstand (class=12)
            x_norm = normalize_coord(ns_x, x_min, x_max)
            z_norm = normalize_coord(ns_z, z_min, z_max)
            y_norm = normalize_coord(0.5, y_min, y_max)
            scene[idx, 22:25] = torch.tensor([x_norm, y_norm, z_norm], device=device)
            sx_norm = normalize_coord(0.25, size_min[0], size_max[0])
            sy_norm = normalize_coord(0.4, size_min[1], size_max[1])
            sz_norm = normalize_coord(0.25, size_min[2], size_max[2])
            scene[idx, 25:28] = torch.tensor([sx_norm, sy_norm, sz_norm], device=device)
            scene[idx, 28:30] = torch.tensor([1.0, 0.0], device=device)
            print(f"  Nightstand at ({ns_x:.2f}, {ns_z:.2f})")
            idx += 1
        
        # Fill empty slots
        for i in range(idx, num_objects):
            scene[i, 21] = 1.0
        
        return scene
    
    # Test 1: Single nightstand on head side (good)
    print("\nTest 1: Single nightstand on head side (left)")
    print("-" * 70)
    scene1 = create_scene(
        bed_config=(0.0, 0.0, 0.0),  # Bed at origin, θ=0 (head at -Z, foot at +Z)
        nightstand_configs=[(-0.9, -1.2)]  # Left side of head
    )
    parsed1 = parse_and_descale_scenes(scene1.unsqueeze(0), num_classes)
    reward1 = compute_night_tables_reward(parsed1, debug=True)
    print(f"Reward: {reward1[0].item():.4f} (should be ~+0.5)")
    
    # Test 2: Two nightstands on both sides of head (excellent)
    print("\nTest 2: Two nightstands on both sides of head (symmetric)")
    print("-" * 70)
    scene2 = create_scene(
        bed_config=(0.0, 0.0, 0.0),
        nightstand_configs=[(-0.9, -1.2), (0.9, -1.2)]  # Left and right of head
    )
    parsed2 = parse_and_descale_scenes(scene2.unsqueeze(0), num_classes)
    reward2 = compute_night_tables_reward(parsed2, debug=True)
    print(f"Reward: {reward2[0].item():.4f} (should be ~+1.0)")
    
    # Test 3: Nightstand on foot side (bad)
    print("\nTest 3: Nightstand on foot side")
    print("-" * 70)
    scene3 = create_scene(
        bed_config=(0.0, 0.0, 0.0),
        nightstand_configs=[(0.9, 1.2)]  # Right side of FOOT (wrong!)
    )
    parsed3 = parse_and_descale_scenes(scene3.unsqueeze(0), num_classes)
    reward3 = compute_night_tables_reward(parsed3, debug=True)
    print(f"Reward: {reward3[0].item():.4f} (should be ~-0.5)")
    
    # Test 4: FIXED - Two nightstands on SAME side (should NOT get symmetric bonus)
    print("\nTest 4: Two nightstands on SAME side (both left - no symmetric bonus)")
    print("-" * 70)
    scene4 = create_scene(
        bed_config=(0.0, 0.0, 0.0),
        nightstand_configs=[(-0.9, -1.2), (-1.2, -1.3)]  # Both on left side
    )
    parsed4 = parse_and_descale_scenes(scene4.unsqueeze(0), num_classes)
    reward4 = compute_night_tables_reward(parsed4, debug=True)
    print(f"Reward: {reward4[0].item():.4f} (should be ~+0.5, NOT +1.0)")
    
    # Test 5: FIXED - Nightstand far to the side (should still work)
    print("\nTest 5: Nightstand far laterally but at head (should work)")
    print("-" * 70)
    scene5 = create_scene(
        bed_config=(0.0, 0.0, 0.0),
        nightstand_configs=[(-1.5, -1.0)]  # Far to the left at head
    )
    parsed5 = parse_and_descale_scenes(scene5.unsqueeze(0), num_classes)
    reward5 = compute_night_tables_reward(parsed5, debug=True)
    print(f"Reward: {reward5[0].item():.4f} (should be ~+0.5)")
    
    print("\n" + "="*70)
    print("✓ All night table tests completed!")
    print("="*70)


if __name__ == "__main__":
    test_night_tables_reward()