import torch
import numpy as np

def point_to_polygon_sdf(point, polygon):
	try:
		from shapely.geometry import Point, Polygon
		p = Point(point)
		poly = Polygon(polygon)
		if poly.contains(p):
			return -poly.exterior.distance(p)
		else:
			return poly.exterior.distance(p)
	except ImportError:
		def winding_number(point, polygon):
			x, y = point
			wn = 0
			for i in range(len(polygon)):
				x0, y0 = polygon[i]
				x1, y1 = polygon[(i+1)%len(polygon)]
				if y0 <= y:
					if y1 > y and (x1-x0)*(y-y0)-(y1-y0)*(x-x0) > 0:
						wn += 1
				else:
					if y1 <= y and (x1-x0)*(y-y0)-(y1-y0)*(x-x0) < 0:
						wn -= 1
			return wn
		min_dist = float('inf')
		for i in range(len(polygon)):
			a = np.array(polygon[i])
			b = np.array(polygon[(i+1)%len(polygon)])
			ab = b - a
			ap = np.array(point) - a
			t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
			closest = a + t * ab
			dist = np.linalg.norm(np.array(point) - closest)
			if dist < min_dist:
				min_dist = dist
		sign = -1 if winding_number(point, polygon) != 0 else 1
		return sign * min_dist

def not_out_of_bound_reward(parsed_scene, floor_polygon):
	positions = parsed_scene["positions"]
	if positions.shape[-1] == 3:
		positions = positions[..., :2]
	N = positions.shape[0]
	floor_polygon = np.array(floor_polygon)
	out_distances = []
	for i in range(N):
		pos = positions[i].detach().cpu().numpy()
		sdf = point_to_polygon_sdf(pos, floor_polygon)
		if sdf > 0:
			out_distances.append(sdf)
	total_out_dist = float(np.sum(out_distances))
	return torch.tensor(total_out_dist, dtype=torch.float32)

# Universal reward wrapper for pipeline compatibility
def compute_not_out_of_bound_reward(parsed_scene, idx, dataset, *args, **kwargs):
	# Accepts extra args for compatibility, but always fetches floor_polygon from dataset
	floor_polygon = dataset.get_floor_polygon_points(idx)
	return not_out_of_bound_reward(parsed_scene, floor_polygon)
# returns the sum of distances of all objects outside the floor plan
# inputs parsed scenes and floor ordered polygon point b, num vertices, 2
# calculate sdf for each floor polygon
# then for each object calculate distance to floor polygon using sdf, assume bbox to be aabb, sizes are already half-extents
import torch
import numpy as np

def point_to_polygon_sdf(point, polygon):
	"""
	Compute signed distance from a 2D point to a polygon.
	Negative if inside, positive if outside.
	"""
	# Use shapely if available, else fallback to manual
	try:
		from shapely.geometry import Point, Polygon
		p = Point(point)
		poly = Polygon(polygon)
		if poly.contains(p):
			return -poly.exterior.distance(p)
		else:
			return poly.exterior.distance(p)
	except ImportError:
		# Manual: min distance to edges, sign by winding number
		def winding_number(point, polygon):
			x, y = point
			wn = 0
			for i in range(len(polygon)):
				x0, y0 = polygon[i]
				x1, y1 = polygon[(i+1)%len(polygon)]
				if y0 <= y:
					if y1 > y and (x1-x0)*(y-y0)-(y1-y0)*(x-x0) > 0:
						wn += 1
				else:
					if y1 <= y and (x1-x0)*(y-y0)-(y1-y0)*(x-x0) < 0:
						wn -= 1
			return wn
		# Min distance to edges
		min_dist = float('inf')
		for i in range(len(polygon)):
			a = np.array(polygon[i])
			b = np.array(polygon[(i+1)%len(polygon)])
			ab = b - a
			ap = np.array(point) - a
			t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0, 1)
			closest = a + t * ab
			dist = np.linalg.norm(np.array(point) - closest)
			if dist < min_dist:
				min_dist = dist
		sign = -1 if winding_number(point, polygon) != 0 else 1
		return sign * min_dist

def compute_not_out_of_bound_reward(parsed_scene, floor_polygon, **kwargs):
	"""
	Returns the sum of distances of all objects outside the floor plan.
	Inputs:
		parsed_scene: dict with keys 'positions', 'sizes' (half-extents), shape (N, 2) and (N, 2)
		floor_polygon: list or array of ordered 2D points (num_vertices, 2)
	Output:
		reward: float (sum of positive distances for objects outside)
	"""
	positions = parsed_scene["positions"]  # (N, 2) or (N, 3)
	sizes = parsed_scene.get("sizes", None)  # (N, 2) or (N, 3)
	if positions.shape[-1] == 3:
		positions = positions[..., :2]  # Use x, y only
	N = positions.shape[0]
	floor_polygon = np.array(floor_polygon)
	out_distances = []
	for i in range(N):
		pos = positions[i].detach().cpu().numpy()
		sdf = point_to_polygon_sdf(pos, floor_polygon)
		if sdf > 0:
			out_distances.append(sdf)
	total_out_dist = float(np.sum(out_distances))
	return torch.tensor(total_out_dist, dtype=torch.float32)

# Example test for consistency
if __name__ == "__main__":
	# Square floor, 2 objects inside, 2 outside
	floor = [[0,0],[0,10],[10,10],[10,0]]
	parsed_scene = {
		"positions": torch.tensor([[5,5],[1,1],[12,5],[5,-2]], dtype=torch.float32),
		"sizes": torch.tensor([[1,1],[1,1],[1,1],[1,1]], dtype=torch.float32)
	}
	reward = compute_not_out_of_bound_reward(parsed_scene, floor)
	print(f"Not out of bound reward: {reward}")
 
 
 ### from claude
 
import numpy as np
from typing import List, Tuple, Dict

class SDFBoundaryChecker:
    """
    Signed Distance Field (SDF) based boundary violation checker.
    
    This class precomputes a 2D grid where each cell stores the signed distance
    to the nearest boundary edge. Positive values = inside, negative = outside.
    This allows O(1) boundary checks for any point in the scene.
    """
    
    def __init__(self, 
                 floor_vertices: List[Tuple[float, float]], 
                 grid_resolution: float = 0.1,
                 world_bounds: Tuple[float, float, float, float] = None):
        """
        Initialize the SDF boundary checker.
        
        Args:
            floor_vertices: List of (x, z) tuples defining the floor polygon.
                           Vertices should be in order (clockwise or counter-clockwise).
                           The polygon is automatically closed.
                           Example: [(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)]
            
            grid_resolution: Size of each grid cell in world units.
                            Smaller = more accurate but more memory/compute.
                            Typical values: 0.05 to 0.2
            
            world_bounds: Optional (min_x, max_x, min_z, max_z) tuple.
                         If None, automatically computed from vertices with padding.
        
        Example:
            # L-shaped room
            floor = [(0,0), (10,0), (10,5), (5,5), (5,10), (0,10)]
            checker = SDFBoundaryChecker(floor, grid_resolution=0.1)
        """
        self.floor_vertices = np.array(floor_vertices, dtype=np.float32)
        self.grid_resolution = grid_resolution
        
        # Auto-compute bounds if not provided
        if world_bounds is None:
            padding = 2.0  # Add padding around the polygon
            min_x = self.floor_vertices[:, 0].min() - padding
            max_x = self.floor_vertices[:, 0].max() + padding
            min_z = self.floor_vertices[:, 1].min() - padding
            max_z = self.floor_vertices[:, 1].max() + padding
            world_bounds = (min_x, max_x, min_z, max_z)
        
        self.world_bounds = world_bounds
        print(f"Precomputing SDF grid (resolution={grid_resolution})...")
        self.sdf_grid, self.x_range, self.z_range = self._compute_sdf_grid()
        print(f"SDF grid ready: {self.sdf_grid.shape[1]}x{self.sdf_grid.shape[0]} cells")
        
    def _point_in_polygon(self, point: np.ndarray) -> bool:
        """
        Ray casting algorithm to test if point is inside polygon.
        Casts a ray from point to infinity and counts edge intersections.
        Odd count = inside, even count = outside.
        """
        x, z = point
        vertices = self.floor_vertices
        n = len(vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, zi = vertices[i]
            xj, zj = vertices[j]
            
            # Check if ray crosses this edge
            if ((zi > z) != (zj > z)) and \
               (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
                inside = not inside
            j = i
            
        return inside
    
    def _distance_to_segment(self, point: np.ndarray, 
                            v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute perpendicular distance from point to line segment.
        Projects point onto segment and clamps to endpoints.
        """
        px, pz = point
        v1x, v1z = v1
        v2x, v2z = v2
        
        # Vector from v1 to v2
        dx = v2x - v1x
        dz = v2z - v1z
        length_sq = dx*dx + dz*dz
        
        # Degenerate case: segment is a point
        if length_sq == 0:
            return np.sqrt((px - v1x)**2 + (pz - v1z)**2)
        
        # Project point onto line, clamped to [0, 1]
        t = ((px - v1x) * dx + (pz - v1z) * dz) / length_sq
        t = np.clip(t, 0, 1)
        
        # Find closest point on segment
        proj_x = v1x + t * dx
        proj_z = v1z + t * dz
        
        return np.sqrt((px - proj_x)**2 + (pz - proj_z)**2)
    
    def _distance_to_polygon(self, point: np.ndarray) -> float:
        """
        Compute minimum distance from point to any polygon edge.
        Returns the shortest perpendicular distance to boundary.
        """
        min_dist = float('inf')
        n = len(self.floor_vertices)
        
        for i in range(n):
            j = (i + 1) % n  # Next vertex (wraps around)
            dist = self._distance_to_segment(point, 
                                             self.floor_vertices[i],
                                             self.floor_vertices[j])
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _compute_sdf_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute the signed distance field grid.
        
        For each grid cell:
        1. Compute distance to nearest boundary edge
        2. Check if point is inside or outside polygon
        3. Store signed distance: positive inside, negative outside
        
        Returns:
            (sdf_grid, x_range, z_range) where sdf_grid[z_idx, x_idx] = signed_distance
        """
        min_x, max_x, min_z, max_z = self.world_bounds
        
        # Create coordinate arrays
        x_range = np.arange(min_x, max_x, self.grid_resolution)
        z_range = np.arange(min_z, max_z, self.grid_resolution)