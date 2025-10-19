"""
PhyScene Rewards Package

This package contains the three main physical guidance constraints for PhyScene:
1. Collision Avoidance Constraint (φcoll)
2. Room Layout Constraint (φlayout) 
3. Reachability Constraint (φreach)

Each constraint is implemented as a standalone module with functions that return loss values.
"""

from .collision_constraint import (
    collision_constraint,
    collision_constraint_with_occupancy,
    multi_scale_collision_constraint
)

from .room_layout_constraint import (
    room_layout_constraint,
    room_boundary_constraint,
    room_center_constraint,
    room_functional_constraint,
    room_density_constraint,
    comprehensive_room_layout_constraint
)

from .reachability_constraint import (
    reachability_constraint,
    multi_agent_reachability_constraint,
    accessibility_constraint,
    comprehensive_reachability_constraint
)

from .common import (
    cal_iou_3d,
    check_articulate,
    draw_2d_gaussian,
    heuristic_distance,
    find_shortest_path,
    map_to_image_coordinate,
    image_to_map_coordinate,
    create_occupancy_map,
    get_region_center,
    calc_path_loss
)

__version__ = "1.0.0"
__author__ = "PhyScene Team"

__all__ = [
    # Collision constraints
    "collision_constraint",
    "collision_constraint_with_occupancy", 
    "multi_scale_collision_constraint",
    
    # Room layout constraints
    "room_layout_constraint",
    "room_boundary_constraint",
    "room_center_constraint", 
    "room_functional_constraint",
    "room_density_constraint",
    "comprehensive_room_layout_constraint",
    
    # Reachability constraints
    "reachability_constraint",
    "multi_agent_reachability_constraint",
    "accessibility_constraint",
    "comprehensive_reachability_constraint",
    
    # Common utilities
    "cal_iou_3d",
    "check_articulate",
    "draw_2d_gaussian",
    "heuristic_distance",
    "find_shortest_path",
    "map_to_image_coordinate",
    "image_to_map_coordinate",
    "create_occupancy_map",
    "get_region_center",
    "calc_path_loss"
]
