import torch
import numpy as np
from typing import Dict, Tuple


def compute_boundary_violation_reward(
    parsed_scene: Dict[str, torch.Tensor],
    floor_polygons: torch.Tensor = None,#list of tensors
    dataset_indices=None,#list of int
    grid_resolution: float = 0.1,
    **kwargs
) -> torch.Tensor:
    """
    Compute boundary violation reward for scene generation using SDF.
    
    Returns negative penalty (sum of violation distances) for objects outside bounds.
    More negative = worse violations. Zero = perfect (all objects inside).
    
    Args:
        parsed_scene: Dictionary with keys:
            - positions: (B, N, 3) - world coordinates [x, y, z]
            - sizes: (B, N, 3) - half-extents [dx, dy, dz]
            - is_empty: (B, N) - boolean mask for empty slots
            - device: device of tensors
        
        floor_polygons: (B, num_vertices, 2) - ordered [x, z] vertices per scene
        
        grid_resolution: SDF grid resolution (default 0.1)
        
        **kwargs: Ignored additional arguments
    
    Returns:
        rewards: (B, 1) - sum of negative violation distances per scene
    """
    # print(f"[Ashok] floor_polygons {floor_polygons}")
    positions = parsed_scene['positions']  # (B, N, 3)
    sizes = parsed_scene['sizes']  # (B, N, 3) - half-extents
    is_empty = parsed_scene['is_empty']  # (B, N)
    device = parsed_scene['device']
    
    B, N = positions.shape[0], positions.shape[1]
    rewards = torch.zeros(B, 1, device=device)
    
    # Process each scene in batch
    for b in range(B):
        floor_verts = floor_polygons[b].cpu().numpy()  # (num_verts, 2)
        
        # Build SDF checker for this scene's floor polygon
        sdf_checker = SDFBoundaryChecker(
            floor_vertices=floor_verts.tolist(),
            grid_resolution=grid_resolution
        )
        
        # Check each object in the scene
        for n in range(N):
            # Skip empty slots
            if is_empty[b, n]:
                continue
            
            # Get object center and size
            obj_pos = positions[b, n].cpu().numpy()  # (3,) [x, y, z]
            obj_size = sizes[b, n].cpu().numpy()  # (3,) [dx, dy, dz] half-extents
            
            # Check 4 corners of object footprint (bottom face)
            # We use XZ plane (ignoring Y/height) for floor collision
            x_center, z_center = obj_pos[0], obj_pos[2]
            dx, dz = obj_size[0], obj_size[2]
            
            # 4 corners of bounding box in XZ plane
            corners = [
                (x_center - dx, z_center - dz),  # bottom-left
                (x_center + dx, z_center - dz),  # bottom-right
                (x_center + dx, z_center + dz),  # top-right
                (x_center - dx, z_center + dz),  # top-left
            ]
            
            # Check violations for all corners
            max_violation = 0.0
            for corner in corners:
                violation = sdf_checker.check_violation(corner)
                max_violation = max(max_violation, violation)
            
            # Accumulate penalty (negative reward)
            rewards[b, 0] -= max_violation
    
    return rewards


class SDFBoundaryChecker:
    """
    Lightweight SDF checker for boundary violations.
    Positive distance = inside, negative = outside.
    """
    
    def __init__(self, floor_vertices, grid_resolution=0.1):
        self.floor_vertices = np.array(floor_vertices, dtype=np.float32)
        self.grid_resolution = grid_resolution
        
        # Auto-compute bounds
        padding = 2.0
        min_x = self.floor_vertices[:, 0].min() - padding
        max_x = self.floor_vertices[:, 0].max() + padding
        min_z = self.floor_vertices[:, 1].min() - padding
        max_z = self.floor_vertices[:, 1].max() + padding
        
        self.world_bounds = (min_x, max_x, min_z, max_z)
        self.sdf_grid, self.x_range, self.z_range = self._compute_sdf_grid()
        
    def check_violation(self, point: Tuple[float, float]) -> float:
        """
        Check if point violates boundary.
        
        Returns:
            0.0 if inside (no violation)
            positive distance if outside (violation magnitude)
        """
        sdf_value = self._query_sdf(point)
        return max(0.0, -sdf_value)  # Only penalize negative (outside) values
    
    def _query_sdf(self, point: Tuple[float, float]) -> float:
        """Query SDF value at point using bilinear interpolation."""
        x, z = point
        min_x, max_x, min_z, max_z = self.world_bounds
        
        # Clamp to grid bounds
        if x < min_x or x >= max_x or z < min_z or z >= max_z:
            return -999.0  # Far outside
        
        # Convert to grid coordinates
        x_idx = (x - min_x) / self.grid_resolution
        z_idx = (z - min_z) / self.grid_resolution
        
        # Bilinear interpolation
        x0, x1 = int(np.floor(x_idx)), int(np.ceil(x_idx))
        z0, z1 = int(np.floor(z_idx)), int(np.ceil(z_idx))
        
        # Clamp indices
        x0 = np.clip(x0, 0, self.sdf_grid.shape[1] - 1)
        x1 = np.clip(x1, 0, self.sdf_grid.shape[1] - 1)
        z0 = np.clip(z0, 0, self.sdf_grid.shape[0] - 1)
        z1 = np.clip(z1, 0, self.sdf_grid.shape[0] - 1)
        
        # Get fractional parts
        fx = x_idx - int(x_idx)
        fz = z_idx - int(z_idx)
        
        # Bilinear weights
        v00 = self.sdf_grid[z0, x0]
        v10 = self.sdf_grid[z0, x1]
        v01 = self.sdf_grid[z1, x0]
        v11 = self.sdf_grid[z1, x1]
        
        v0 = v00 * (1 - fx) + v10 * fx
        v1 = v01 * (1 - fx) + v11 * fx
        
        return v0 * (1 - fz) + v1 * fz
    
    def _point_in_polygon(self, point: np.ndarray) -> bool:
        """Ray casting algorithm."""
        x, z = point
        vertices = self.floor_vertices
        n = len(vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, zi = vertices[i]
            xj, zj = vertices[j]
            
            if ((zi > z) != (zj > z)) and \
               (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def _distance_to_segment(self, point: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
        """Distance from point to line segment."""
        px, pz = point
        v1x, v1z = v1
        v2x, v2z = v2
        
        dx = v2x - v1x
        dz = v2z - v1z
        length_sq = dx*dx + dz*dz
        
        if length_sq == 0:
            return np.sqrt((px - v1x)**2 + (pz - v1z)**2)
        
        t = ((px - v1x) * dx + (pz - v1z) * dz) / length_sq
        t = np.clip(t, 0, 1)
        
        proj_x = v1x + t * dx
        proj_z = v1z + t * dz
        
        return np.sqrt((px - proj_x)**2 + (pz - proj_z)**2)
    
    def _distance_to_polygon(self, point: np.ndarray) -> float:
        """Minimum distance to polygon boundary."""
        min_dist = float('inf')
        n = len(self.floor_vertices)
        
        for i in range(n):
            j = (i + 1) % n
            dist = self._distance_to_segment(point, 
                                             self.floor_vertices[i],
                                             self.floor_vertices[j])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _compute_sdf_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Precompute SDF grid."""
        min_x, max_x, min_z, max_z = self.world_bounds
        
        x_range = np.arange(min_x, max_x, self.grid_resolution)
        z_range = np.arange(min_z, max_z, self.grid_resolution)
        
        sdf_grid = np.zeros((len(z_range), len(x_range)), dtype=np.float32)
        
        for i, z in enumerate(z_range):
            for j, x in enumerate(x_range):
                point = np.array([x, z])
                dist = self._distance_to_polygon(point)
                is_inside = self._point_in_polygon(point)
                sdf_grid[i, j] = dist if is_inside else -dist
        
        return sdf_grid, x_range, z_range