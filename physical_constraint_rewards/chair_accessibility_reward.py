"""
lets impement this rward, this dataset forward is the direction as if we are sitting behind the object. eg. front of bed(orientation 0 rad) we are sitting behind headboard and looking at bed infront of us. so how do we implemetn this reward
Chair Accessibility Reward - Ensures chairs/sofas face useful directions.

Forward direction is defined as "the direction you'd look when sitting in the object":
- Chair/Sofa (θ=0): Forward is where seated person looks (away from backrest)
- This reward encourages chairs to face tables or open space, NOT walls

Penalty cases:
- Chair facing wall within close distance (< 0.5m to wall)
- Chair facing nothing useful (isolated in corner)

Reward cases:
- Chair facing a table (dining setup)
- Chair facing open space (conversation area, TV viewing)

Room bounds (from dataset):
- X: [-2.76, 2.78]  (width ~5.54m)
- Z: [-2.75, 2.82]  (depth ~5.57m)
- Y: [0.045, 3.62]  (height, but we care about XZ plane for facing)
"""

import torch
import torch.nn.functional as F


def compute_facing_direction(theta):
    """
    Compute forward direction vector from orientation angle.
    
    Forward = direction you'd look when sitting/using the object
    For θ=0: forward is +Z direction (north)
    Rotates counter-clockwise around Y-axis
    
    Args:
        theta: (B, N) tensor of orientation angles in radians
    
    Returns:
        forward: (B, N, 2) tensor of unit vectors in XZ plane [dx, dz]
    """
    # Forward direction in XZ plane (Y-axis is up)
    # θ=0 → forward=[0, 1] (+Z direction)
    # θ=π/2 → forward=[-1, 0] (-X direction)
    dx = -torch.sin(theta)  # X component
    dz = torch.cos(theta)   # Z component
    
    forward = torch.stack([dx, dz], dim=-1)  # (B, N, 2)
    return forward


def check_facing_wall(positions, forward_dirs, room_bounds, ray_distance=1.0, debug=False):
    """
    Check if forward direction hits a wall within ray_distance.
    
    Cast ray from object position along forward direction.
    Check intersection with room walls (4 walls in XZ plane).
    
    Args:
        positions: (B, N, 3) tensor of [x, y, z] positions
        forward_dirs: (B, N, 2) tensor of forward directions in XZ plane
        room_bounds: dict with 'x_min', 'x_max', 'z_min', 'z_max'
        ray_distance: Maximum distance to check for wall (default 1.0m)
        debug: If True, print debug information
    
    Returns:
        facing_wall: (B, N) boolean tensor, True if facing wall within distance
    """
    B, N = positions.shape[:2]
    device = positions.device
    
    # Extract XZ positions
    x_pos = positions[:, :, 0]  # (B, N)
    z_pos = positions[:, :, 2]  # (B, N)
    
    # Forward direction components
    dx = forward_dirs[:, :, 0]  # (B, N)
    dz = forward_dirs[:, :, 1]  # (B, N)
    
    # Room wall positions
    x_min, x_max = room_bounds['x_min'], room_bounds['x_max']
    z_min, z_max = room_bounds['z_min'], room_bounds['z_max']
    
    if debug:
        print(f"  Room bounds: X[{x_min:.2f}, {x_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
        print(f"  Ray distance: {ray_distance}")
    
    # Initialize: not facing wall
    facing_wall = torch.zeros(B, N, dtype=torch.bool, device=device)
    
    # Check each wall (4 walls in XZ plane)
    eps = 1e-6  # Avoid division by zero
    
    # Wall 1: X = x_min (west wall)
    # Ray: [x_pos, z_pos] + t * [dx, dz]
    # Intersection: x_pos + t * dx = x_min → t = (x_min - x_pos) / dx
    t_x_min = (x_min - x_pos) / (dx + eps)
    valid_x_min = (dx < -eps) & (t_x_min > 0) & (t_x_min < ray_distance)
    facing_wall |= valid_x_min
    
    if debug:
        for b in range(B):
            for n in range(N):
                if valid_x_min[b, n]:
                    print(f"    Object ({b},{n}): Facing WEST wall, t={t_x_min[b,n]:.2f}m")
    
    # Wall 2: X = x_max (east wall)
    t_x_max = (x_max - x_pos) / (dx + eps)
    valid_x_max = (dx > eps) & (t_x_max > 0) & (t_x_max < ray_distance)
    facing_wall |= valid_x_max
    
    if debug:
        for b in range(B):
            for n in range(N):
                if valid_x_max[b, n]:
                    print(f"    Object ({b},{n}): Facing EAST wall, t={t_x_max[b,n]:.2f}m")
    
    # Wall 3: Z = z_min (south wall)
    t_z_min = (z_min - z_pos) / (dz + eps)
    valid_z_min = (dz < -eps) & (t_z_min > 0) & (t_z_min < ray_distance)
    facing_wall |= valid_z_min
    
    if debug:
        for b in range(B):
            for n in range(N):
                if valid_z_min[b, n]:
                    print(f"    Object ({b},{n}): Facing SOUTH wall, t={t_z_min[b,n]:.2f}m")
                elif dz[b, n] < -eps:
                    # Facing south but not hitting wall
                    print(f"    Object ({b},{n}): Facing SOUTH but too far: t={t_z_min[b,n]:.2f}m > {ray_distance}m")
    
    # Wall 4: Z = z_max (north wall)
    t_z_max = (z_max - z_pos) / (dz + eps)
    valid_z_max = (dz > eps) & (t_z_max > 0) & (t_z_max < ray_distance)
    facing_wall |= valid_z_max
    
    if debug:
        for b in range(B):
            for n in range(N):
                if valid_z_max[b, n]:
                    print(f"    Object ({b},{n}): Facing NORTH wall, t={t_z_max[b,n]:.2f}m")
    
    return facing_wall


def check_facing_table(chair_positions, chair_forward, table_positions, table_sizes,
                       max_distance=1.5, cone_angle=60.0):
    """
    Check if chair is facing a table within reasonable distance and angle.
    
    A chair "faces" a table if:
    1. Table center is within max_distance
    2. Table is in the forward cone (within ±cone_angle degrees)
    
    Args:
        chair_positions: (B, N_chairs, 3) positions of chairs
        chair_forward: (B, N_chairs, 2) forward directions in XZ plane
        table_positions: (B, N_tables, 3) positions of tables
        table_sizes: (B, N_tables, 3) half-extents of tables
        max_distance: Maximum distance to consider (default 1.5m)
        cone_angle: Half-angle of forward cone in degrees (default 60°)
    
    Returns:
        facing_table: (B, N_chairs) boolean tensor
    """
    B, N_chairs = chair_positions.shape[:2]
    N_tables = table_positions.shape[1]
    device = chair_positions.device
    
    if N_tables == 0:
        # No tables in scene
        return torch.zeros(B, N_chairs, dtype=torch.bool, device=device)
    
    # Extract XZ coordinates
    chair_xz = chair_positions[:, :, [0, 2]]  # (B, N_chairs, 2)
    table_xz = table_positions[:, :, [0, 2]]  # (B, N_tables, 2)
    
    # Compute pairwise vectors from chairs to tables
    # (B, N_chairs, 1, 2) - (B, 1, N_tables, 2) → (B, N_chairs, N_tables, 2)
    chair_to_table = table_xz.unsqueeze(1) - chair_xz.unsqueeze(2)
    
    # Distances in XZ plane
    distances = torch.norm(chair_to_table, dim=-1)  # (B, N_chairs, N_tables)
    
    # Normalize chair-to-table vectors
    chair_to_table_norm = F.normalize(chair_to_table, dim=-1, eps=1e-8)
    
    # Expand chair forward direction for broadcasting
    chair_forward_exp = chair_forward.unsqueeze(2)  # (B, N_chairs, 1, 2)
    
    # Compute dot product: how aligned is table with forward direction
    # dot = cos(angle)
    dot_product = (chair_to_table_norm * chair_forward_exp).sum(dim=-1)  # (B, N_chairs, N_tables)
    
    # Convert cone angle to radians and compute threshold
    cone_rad = torch.tensor(cone_angle * 3.14159 / 180.0, device=device)
    cos_threshold = torch.cos(cone_rad)
    
    # Chair faces table if:
    # 1. Within max_distance
    # 2. Within forward cone (dot > cos_threshold)
    facing_any_table = ((distances < max_distance) & (dot_product > cos_threshold)).any(dim=2)
    
    return facing_any_table  # (B, N_chairs)


def compute_chair_accessibility_reward(parsed_scene, wall_penalty=-1.0, 
                                      table_bonus=0.5, open_space_bonus=0.2, debug=False, **kwargs):
    """
    Calculate reward for chair/sofa accessibility and facing direction.
    
    Encourages:
    - Chairs facing tables (dining, work setups)
    - Chairs facing open space (conversation, TV)
    
    Penalizes:
    - Chairs facing walls (unusable, awkward)
    
    Args:
        parsed_scene: Dict from parse_and_descale_scenes()
        wall_penalty: Penalty for facing wall (default -1.0)
        table_bonus: Reward for facing table (default +0.5)
        open_space_bonus: Reward for facing open space (default +0.2)
        **kwargs: Additional arguments
    
    Returns:
        rewards: (B,) tensor with per-scene rewards
    """
    positions = parsed_scene['positions']  # (B, N, 3)
    sizes = parsed_scene['sizes']  # (B, N, 3) - HALF-EXTENTS
    orientations_cos_sin = parsed_scene['orientations']  # (B, N, 2) [cos, sin]
    object_indices = parsed_scene['object_indices']  # (B, N)
    is_empty = parsed_scene['is_empty']  # (B, N)
    
    B, N = positions.shape[:2]
    device = positions.device
    
    # Convert cos/sin back to radians
    cos_theta = orientations_cos_sin[:, :, 0]
    sin_theta = orientations_cos_sin[:, :, 1]
    orientations = torch.atan2(sin_theta, cos_theta)  # (B, N) in radians
    
    # Room bounds (from dataset statistics)
    room_bounds = {
        'x_min': -2.76,
        'x_max': 2.78,
        'z_min': -2.75,
        'z_max': 2.82,
    }
    
    # Identify seating furniture (chairs, sofas, benches)
    # Class IDs (from idx_to_labels): chair=4, armchair=0, sofa=16, stool=17
    seating_classes = [0, 4, 16, 17]  # armchair, chair, sofa, stool
    
    is_seating = torch.zeros(B, N, dtype=torch.bool, device=device)
    for cls_id in seating_classes:
        is_seating |= (object_indices == cls_id)
    
    # Filter out empty slots
    is_seating = is_seating & (~is_empty)
    
    # Identify tables (coffee_table=6, desk=7, dressing_table=10, table=18)
    table_classes = [6, 7, 10, 18]
    is_table = torch.zeros(B, N, dtype=torch.bool, device=device)
    for cls_id in table_classes:
        is_table |= (object_indices == cls_id)
    is_table = is_table & (~is_empty)
    
    # Compute forward directions for all objects
    forward_dirs = compute_facing_direction(orientations)  # (B, N, 2)
    
    # Check which chairs face walls
    facing_wall = check_facing_wall(positions, forward_dirs, room_bounds, ray_distance=1.0, debug=debug)
    
    # Extract table information for facing check
    # Get positions and sizes of tables only
    table_positions_list = []
    table_sizes_list = []
    for b in range(B):
        table_mask = is_table[b]
        if table_mask.any():
            table_positions_list.append(positions[b, table_mask])
            table_sizes_list.append(sizes[b, table_mask])
        else:
            # No tables - add dummy
            table_positions_list.append(torch.zeros(0, 3, device=device))
            table_sizes_list.append(torch.zeros(0, 3, device=device))
    
    # Pad to same size for batching
    max_tables = max(tp.shape[0] for tp in table_positions_list)
    if max_tables == 0:
        max_tables = 1  # At least 1 for tensor creation
    
    table_positions_batch = torch.zeros(B, max_tables, 3, device=device)
    table_sizes_batch = torch.zeros(B, max_tables, 3, device=device)
    for b in range(B):
        n_tables = table_positions_list[b].shape[0]
        if n_tables > 0:
            table_positions_batch[b, :n_tables] = table_positions_list[b]
            table_sizes_batch[b, :n_tables] = table_sizes_list[b]
    
    # Check which chairs face tables
    facing_table = check_facing_table(positions, forward_dirs, 
                                     table_positions_batch, table_sizes_batch)
    
    # Compute rewards per scene
    rewards = torch.zeros(B, device=device)
    
    for b in range(B):
        seating_mask = is_seating[b]
        n_seating = seating_mask.sum().item()
        
        if debug:
            print(f"\nScene {b}: Found {n_seating} seating furniture")
            print(f"  Seating indices: {torch.where(seating_mask)[0].tolist()}")
            print(f"  Table indices: {torch.where(is_table[b])[0].tolist()}")
        
        if n_seating == 0:
            # No seating furniture - neutral reward
            continue
        
        # Count violations and good placements
        n_facing_wall = (facing_wall[b] & seating_mask).sum().float()
        n_facing_table = (facing_table[b] & seating_mask).sum().float()
        
        # Chairs facing open space = not facing wall and not facing table
        n_facing_open = ((~facing_wall[b]) & (~facing_table[b]) & seating_mask).sum().float()
        
        if debug:
            print(f"  Facing wall: {n_facing_wall.item()}")
            print(f"  Facing table: {n_facing_table.item()}")
            print(f"  Facing open: {n_facing_open.item()}")
            # Print individual chair info
            for idx in torch.where(seating_mask)[0]:
                pos = positions[b, idx]
                fwd = forward_dirs[b, idx]
                print(f"    Chair {idx}: pos=({pos[0]:.2f}, {pos[2]:.2f}), "
                      f"forward=({fwd[0]:.2f}, {fwd[1]:.2f}), "
                      f"wall={facing_wall[b, idx].item()}, "
                      f"table={facing_table[b, idx].item()}")
        
        # Compute reward components
        wall_reward = n_facing_wall * wall_penalty
        table_reward = n_facing_table * table_bonus
        open_reward = n_facing_open * open_space_bonus
        
        # Total reward for this scene (normalized by number of seating items)
        rewards[b] = (wall_reward + table_reward + open_reward) / n_seating
    
    return rewards


def test_chair_accessibility_reward():
    """Test cases for chair accessibility reward."""
    print("\n" + "="*70)
    print("Testing Chair Accessibility Reward")
    print("="*70)
    
    device = 'cpu'
    num_classes = 22
    num_objects = 12
    
    def create_test_scene(chair_configs, table_configs=None):
        """
        Create test scene with chairs and optional tables.
        
        chair_configs: list of (x, z, theta_deg) for each chair (in WORLD coordinates)
        table_configs: list of (x, z) for each table (in WORLD coordinates)
        
        This function NORMALIZES the coordinates to [-1, 1] before storing.
        """
        scene = torch.zeros(num_objects, 30, device=device)
        idx = 0
        
        # Room bounds for normalization
        x_min, x_max = -2.7625005, 2.7784417
        z_min, z_max = -2.75275, 2.8185427
        y_min, y_max = 0.045, 3.6248395
        
        def normalize_coord(val, min_val, max_val):
            """Normalize to [-1, 1]"""
            return 2 * (val - min_val) / (max_val - min_val) - 1
        
        # Add chairs
        for x, z, theta_deg in chair_configs:
            theta_rad = torch.tensor(theta_deg * 3.14159 / 180.0, device=device)
            scene[idx, 4] = 1.0  # Chair class (id=4)
            # Normalize positions to [-1, 1]
            x_norm = normalize_coord(x, x_min, x_max)
            z_norm = normalize_coord(z, z_min, z_max)
            y_norm = normalize_coord(1.0, y_min, y_max)
            scene[idx, 22:25] = torch.tensor([x_norm, y_norm, z_norm], device=device)
            # Normalize sizes to [-1, 1] (half-extents: 0.3m, 0.4m, 0.3m)
            size_min = torch.tensor([0.03998289, 0.02000002, 0.012772], device=device)
            size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)
            sx_norm = normalize_coord(0.3, size_min[0], size_max[0])
            sy_norm = normalize_coord(0.4, size_min[1], size_max[1])
            sz_norm = normalize_coord(0.3, size_min[2], size_max[2])
            scene[idx, 25:28] = torch.tensor([sx_norm, sy_norm, sz_norm], device=device)
            scene[idx, 28] = torch.cos(theta_rad)  # cos(theta)
            scene[idx, 29] = torch.sin(theta_rad)  # sin(theta)
            print(f"  Adding chair at world=({x:.2f}, {z:.2f}), norm=({x_norm:.2f}, {z_norm:.2f}), θ={theta_deg}° (forward dir: ({-torch.sin(theta_rad).item():.2f}, {torch.cos(theta_rad).item():.2f}))")
            idx += 1
        
        # Add tables if provided
        if table_configs:
            for x, z in table_configs:
                scene[idx, 18] = 1.0  # Table class (id=18)
                x_norm = normalize_coord(x, x_min, x_max)
                z_norm = normalize_coord(z, z_min, z_max)
                y_norm = normalize_coord(0.75, y_min, y_max)
                scene[idx, 22:25] = torch.tensor([x_norm, y_norm, z_norm], device=device)
                size_min = torch.tensor([0.03998289, 0.02000002, 0.012772], device=device)
                size_max = torch.tensor([2.8682, 1.770065, 1.698315], device=device)
                sx_norm = normalize_coord(0.6, size_min[0], size_max[0])
                sy_norm = normalize_coord(0.05, size_min[1], size_max[1])
                sz_norm = normalize_coord(0.4, size_min[2], size_max[2])
                scene[idx, 25:28] = torch.tensor([sx_norm, sy_norm, sz_norm], device=device)
                scene[idx, 28:30] = torch.tensor([1.0, 0.0], device=device)  # theta=0
                print(f"  Adding table at world=({x:.2f}, {z:.2f}), norm=({x_norm:.2f}, {z_norm:.2f})")
                idx += 1
        
        # Fill remaining with empty
        for i in range(idx, num_objects):
            scene[i, 21] = 1.0  # Empty class
        
        return scene
    
    from physical_constraint_rewards.commons import parse_and_descale_scenes
    
    # Test 1: Chair facing wall (bad)
    print("\nTest 1: Chair facing wall")
    print("-" * 70)
    scene1 = create_test_scene([
        (2.5, 0.0, 270.0),  # Chair near east wall, facing east (θ=270° → facing +X)
    ])
    parsed1 = parse_and_descale_scenes(scene1.unsqueeze(0), num_classes)
    reward1 = compute_chair_accessibility_reward(parsed1)
    print(f"Reward: {reward1[0].item():.4f} (expected: -1.0) ✓")
    
    # Test 2: Chair facing table (good)
    print("\nTest 2: Chair facing table")
    print("-" * 70)
    scene2 = create_test_scene(
        chair_configs=[
            (0.0, -0.5, 0.0),  # Chair closer, facing north (θ=0° → facing +Z)
        ],
        table_configs=[
            (0.0, 0.5),  # Table at +Z (in front of chair, 1m away)
        ]
    )
    parsed2 = parse_and_descale_scenes(scene2.unsqueeze(0), num_classes)
    reward2 = compute_chair_accessibility_reward(parsed2)
    print(f"Reward: {reward2[0].item():.4f} (expected: +0.5) ✓")
    
    # Test 3: Chair facing open space (good)
    print("\nTest 3: Chair facing open space")
    print("-" * 70)
    scene3 = create_test_scene([
        (0.0, 0.0, 0.0),  # Chair in center, facing open space
    ])
    parsed3 = parse_and_descale_scenes(scene3.unsqueeze(0), num_classes)
    reward3 = compute_chair_accessibility_reward(parsed3)
    print(f"Reward: {reward3[0].item():.4f} (expected: +0.2) ✓")
    
    # Test 4: Mixed scene
    print("\nTest 4: Mixed scene (2 chairs: 1 good, 1 bad)")
    print("-" * 70)
    scene4 = create_test_scene(
        chair_configs=[
            (0.0, 0.0, 0.0),    # Chair 1: facing table (good)
            (2.5, 0.0, 270.0),  # Chair 2: facing east wall (bad, θ=270° → facing +X)
        ],
        table_configs=[
            (0.0, 0.8),  # Table in front of chair 1
        ]
    )
    parsed4 = parse_and_descale_scenes(scene4.unsqueeze(0), num_classes)
    reward4 = compute_chair_accessibility_reward(parsed4)
    print(f"Reward: {reward4[0].item():.4f} (expected: -0.25) ✓")
    
    print("\n" + "="*70)
    print("✓ All chair accessibility tests completed!")
    print("="*70)


if __name__ == "__main__":
    test_chair_accessibility_reward()