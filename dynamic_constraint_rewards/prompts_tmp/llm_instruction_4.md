
    # TASK: Assigning inportance weights to each of dynamic and universal reward components.

    You are an expert in 3D scene generation, interior design, and reinforcement learning.
    Your task is to analyze the user prompt, final constraints, final dynamic and universal reward functions. Then, return the weights to be applied to each of the rewards while training the reinforcement learning model.

    Now, as an reinforcement learning expert in reward shaping if any of the reward functions conflict then according to the desired behaviour as specified in the user prompt, return the weights to be applied to each of the rewards such that the final reward value will be the most suitable while training the reinforcement learning model.

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
    

    Here is the dataset information in JSON format about the specific room type: livingroom you will be working on:
    ```json
    {'room_type': 'livingroom', 'total_scenes': 2926, 'class_frequencies': {'dining_chair': 0.25492085340674464, 'pendant_lamp': 0.13282863041982107, 'coffee_table': 0.08616655196145905, 'corner_side_table': 0.07240192704748796, 'dining_table': 0.06951135581555402, 'tv_stand': 0.06221610461114935, 'multi_seat_sofa': 0.05299380591878871, 'armchair': 0.048313833448038544, 'console_table': 0.037026841018582245, 'lounge_chair': 0.03234686854783207, 'stool': 0.0264280798348245, 'cabinet': 0.023124569855471438, 'bookshelf': 0.02202339986235375, 'loveseat_sofa': 0.020922229869236062, 'ceiling_lamp': 0.018169304886441844, 'wine_cabinet': 0.012112869924294563, 'l_shaped_sofa': 0.01032346868547832, 'round_end_table': 0.0057811424638678595, 'shelf': 0.0035788024776324846, 'chinese_chair': 0.0031658637302133517, 'wardrobe': 0.0027529249827942187, 'chaise_longue_sofa': 0.0011011699931176876, 'desk': 0.0009635237439779766, 'lazy_sofa': 0.0008258774948382657}, 'furniture_counts': {'dining_chair': 1852, 'pendant_lamp': 965, 'coffee_table': 626, 'corner_side_table': 526, 'dining_table': 505, 'tv_stand': 452, 'multi_seat_sofa': 385, 'armchair': 351, 'console_table': 269, 'lounge_chair': 235, 'stool': 192, 'cabinet': 168, 'bookshelf': 160, 'loveseat_sofa': 152, 'ceiling_lamp': 132, 'wine_cabinet': 88, 'l_shaped_sofa': 75, 'round_end_table': 42, 'shelf': 26, 'chinese_chair': 23, 'wardrobe': 20, 'chaise_longue_sofa': 8, 'desk': 7, 'lazy_sofa': 6}, 'idx_to_labels': {0: 'armchair', 1: 'bookshelf', 2: 'cabinet', 3: 'ceiling_lamp', 4: 'chaise_longue_sofa', 5: 'chinese_chair', 6: 'coffee_table', 7: 'console_table', 8: 'corner_side_table', 9: 'desk', 10: 'dining_chair', 11: 'dining_table', 12: 'l_shaped_sofa', 13: 'lazy_sofa', 14: 'lounge_chair', 15: 'loveseat_sofa', 16: 'multi_seat_sofa', 17: 'pendant_lamp', 18: 'round_end_table', 19: 'shelf', 20: 'stool', 21: 'tv_stand', 22: 'wardrobe', 23: 'wine_cabinet', 24: 'empty'}, 'num_classes_with_empty': 25, 'num_classes_without_empty': 24, 'max_objects': 21}
    ```

    ### Scene Representation
    
    A 3D scene is represented in batch format (parsed_scenes) a dictionary with the following keys and PyTorch tensors as values:
        - `positions`: (B, N, 3) - Object centroids in meters (x, y, z)
        - `sizes`: (B, N, 3) - Half-extents (sx/2, sy/2, sz/2)
        - `object_indices`: (B, N) - Class indices [0, num_classes-1]
        - `one_hot`: (B, N, num_classes) - One-hot encoded classes
        - `is_empty`: (B, N) - Boolean mask (True = empty slot)
        - `orientations`: (B, N, 2) - [cos(θ), sin(θ)] for z-rotation
        - `device`: torch.device
        Where:
            - B = Batch size
            - N = Max objects per scene
    


    ### The baseline model is already trained on some universal constraints and they are used as part of reward functions as well for regularization,


    ## YOUR TASK
    Analyze the original user prompt, final constraints, final dynamic reward functions and universal reward functions, then provide a comprehensive JSON response with the following structure.

    It should be noted that each reward components is converted to the range [-1, 1] before applying the weighted sum so the weights should purely be based on the importance of the rewards. This task is aimed to reduce conflicting rewards because some dynamic reward may try to conflict with these universal ones.

    ---
    Only return the following JSON response (nothing else), follow this structure strictly:

    ```json
    {
    "importance_weights": {
        "reward_name1": weight1(float),
        "reward_name2": weight2(float),
        ...
        "reward_namen": weightn(float)
    }
    }
    ```

    