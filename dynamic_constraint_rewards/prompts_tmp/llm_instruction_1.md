
    # TASK: Constraint Decomposition for 3D Scene Generation

    You are an expert in 3D scene generation, interior design, and reinforcement learning. Your task is to analyze a user prompt and decompose it into verifiable constraints with Python reward functions.

    ## CONTEXT

    ### Dataset: 3D-FRONT
    
    The dataset being used is 3D-FRONT which uses 3D-FUTURE dataset for furniture models. 3D-FRONT is a collection of synthetic, high-quality 3D indoor scenes, highlighted by professionally and distinctively designed layouts.

    In this dataset, the following facts are important to know:

    ## Coordinate System
    - Y-axis: Vertical (up direction)
    - XZ-plane: Floor plane
    - Units: Meters (world coordinates, unnormalized)
    - Empty slots: Have index (num_classes-1), near-zero size/position

    # Important Facts about 3D-FRONT dataset
    - Ceiling objects are at y ≈ ceiling_height (typically 2.8m)
    - Floor objects have y ≈ object_height/2
    - Ignore empty slots (is_empty == True) in calculations
    

    Note: While generating constraints, no need to verify these facts with you constraints, focus on the constraints other than these facts.

    In this task, you are provided with a user prompt and a dataset context. Your task is to decompose the user prompt into verifiable constraints with Python reward functions.

    Here is the dataset information in JSON format about the specific room type: livingroom you will be working on:
    ```json
    {'room_type': 'livingroom', 'total_scenes': 2926, 'class_frequencies': {'dining_chair': 0.25492085340674464, 'pendant_lamp': 0.13282863041982107, 'coffee_table': 0.08616655196145905, 'corner_side_table': 0.07240192704748796, 'dining_table': 0.06951135581555402, 'tv_stand': 0.06221610461114935, 'multi_seat_sofa': 0.05299380591878871, 'armchair': 0.048313833448038544, 'console_table': 0.037026841018582245, 'lounge_chair': 0.03234686854783207, 'stool': 0.0264280798348245, 'cabinet': 0.023124569855471438, 'bookshelf': 0.02202339986235375, 'loveseat_sofa': 0.020922229869236062, 'ceiling_lamp': 0.018169304886441844, 'wine_cabinet': 0.012112869924294563, 'l_shaped_sofa': 0.01032346868547832, 'round_end_table': 0.0057811424638678595, 'shelf': 0.0035788024776324846, 'chinese_chair': 0.0031658637302133517, 'wardrobe': 0.0027529249827942187, 'chaise_longue_sofa': 0.0011011699931176876, 'desk': 0.0009635237439779766, 'lazy_sofa': 0.0008258774948382657}, 'furniture_counts': {'dining_chair': 1852, 'pendant_lamp': 965, 'coffee_table': 626, 'corner_side_table': 526, 'dining_table': 505, 'tv_stand': 452, 'multi_seat_sofa': 385, 'armchair': 351, 'console_table': 269, 'lounge_chair': 235, 'stool': 192, 'cabinet': 168, 'bookshelf': 160, 'loveseat_sofa': 152, 'ceiling_lamp': 132, 'wine_cabinet': 88, 'l_shaped_sofa': 75, 'round_end_table': 42, 'shelf': 26, 'chinese_chair': 23, 'wardrobe': 20, 'chaise_longue_sofa': 8, 'desk': 7, 'lazy_sofa': 6}, 'idx_to_labels': {0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chaise_longue_sofa', 5: 'chinese_chair', 6: 'coffee_table', 7: 'console_table', 8: 'corner_side_table', 9: 'desk', 10: 'dining_chair', 11: 'dining_table', 12: 'l_shaped_sofa', 13: 'lazy_sofa', 14: 'lounge_chair', 15: 'loveseat_sofa', 16: 'multi_seat_sofa', 17: 'pendant_lamp', 18: 'round_end_table', 19: 'shelf', 20: 'stool', 21: 'tv_stand', 22: 'wardrobe', 23: 'wine_cabinet', 24: 'empty'}, 'num_classes_with_empty': 25, 'num_classes_without_empty': 24, 'max_objects': 21}
    ```

    Also, the baseline model is already trained on some universal constraints, so you do not need to consider these constraints while generating new ones. The universal constraints are:
    ```json
    {'non_penetration': {'function': 'compute_non_penetration_reward', 'description': '\n    Calculate reward based on non-penetration constraint using penetration depth.\n\n    Following the approach from original authors: reward = sum of negative signed distances.\n    When objects overlap, we get positive penetration depth, so reward is negative.\n\n    Args:\n        parsed_scene: Dict returned by parse_and_descale_scenes()\n\n    Returns:\n        rewards: Tensor of shape (B,) with non-penetration rewards for each scene\n    '}, 'not_out_of_bound': {'function': 'compute_boundary_violation_reward', 'description': "\n    Compute boundary violation reward using cached SDF grids.\n\n    **IMPORTANT**: Call `precompute_sdf_cache()` once before training to generate cache!\n\n    Args:\n        parsed_scene: Dictionary with positions, sizes, is_empty, device\n        floor_polygons: (B, num_vertices, 2) - only needed if cache doesn't exist\n        indices: (B,) - scene indices for SDF lookup\n        grid_resolution: SDF grid resolution\n        sdf_cache_dir: Directory containing cached SDF grids\n\n    Returns:\n        rewards: (B, 1) - sum of negative violation distances per scene\n    "}, 'accessibility': {'function': 'compute_accessibility_reward', 'description': "\n    Compute accessibility reward using cached floor grids or computing on-the-fly.\n\n    Returns dict with 3 components:\n    - coverage_ratio: [0, 1] - fraction of floor reachable from largest region\n    - num_regions: [1, ∞) - number of disconnected regions\n    - avg_clearance: meters - average distance to nearest obstacle in reachable area\n\n    Args:\n        parsed_scenes: Dictionary with positions, sizes, is_empty, device, object_types\n        floor_polygons: (B, num_vertices, 2) - floor polygon vertices\n        is_val: Whether this is validation split\n        indices: (B,) - scene indices for cache lookup\n        accessibility_cache: Pre-loaded AccessibilityCache instance (optional)\n        grid_resolution: Grid resolution in meters (default 0.2m = 20cm)\n        agent_radius: Agent radius in meters (default 0.15m = 15cm)\n        save_viz: Whether to save visualization images\n        viz_dir: Directory to save visualizations\n\n    Returns:\n        Dictionary with:\n        - 'coverage_ratio': (B,) - reachable area ratio [0, 1]\n        - 'num_regions': (B,) - number of disconnected regions [1, ∞)\n        - 'avg_clearance': (B,) - average clearance in meters\n    "}, 'gravity_following': {'function': 'compute_gravity_following_reward', 'description': '\n    Calculate gravity-following reward based on how close objects are to the ground.\n\n    Objects should rest on the floor (y_min ≈ 0), except for ceiling objects.\n    Only penalizes objects that are MORE than tolerance away from the floor(both sinking and floating cases).\n\n    Args:\n        parsed_scene: Dict returned by parse_and_descale_scenes()\n        tolerance: Distance threshold in meters (default 0.01m = 1cm)\n\n    Returns:\n        rewards: Tensor of shape (B,) with gravity-following rewards\n    '}}
    ```

    ## YOUR TASK

    Analyze the user prompt and provide a comprehensive JSON response with the following structure:

    ### 1. CONSTRAINT DECOMPOSITION

    Generate ALL constraints needed to satisfy the prompt strictly in following format.

    ```json
    {
    "constraints": [
        {
        "id": "C1",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        },
        {
        "id": "C2",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        },
        ...
        {
        "id": "Cn",
        "name": "descriptive_snake_case_name",
        "description": "Clear description of what this checks"
        }
    ]
    }
    ```
    