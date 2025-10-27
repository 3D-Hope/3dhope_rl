"""
Script to compute reward statistics from ground truth dataset.

This script loads scenes from the ground truth dataset and computes statistics 
for multiple reward functions.
"""

import json
import os

from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from omegaconf import DictConfig

from steerable_scene_generation.datasets.custom_scene import CustomDataset
from universal_constraint_rewards.commons import parse_and_descale_scenes
from universal_constraint_rewards.not_out_of_bound_reward import precompute_sdf_cache, SDFCache


def get_reward_stats_from_dataset(
    reward_functions: Dict[str, Callable],
    config: DictConfig = None,
    num_scenes: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Load scenes from ground truth dataset and compute reward statistics.

    Args:
        reward_functions: Dict mapping reward function names to callable functions
                         Each function should take a scene dict and return a scalar reward [0, 1]
        config: Configuration object containing dataset paths
        num_scenes: Number of scenes to load from dataset (None = all scenes)

    Returns:
        Dict mapping reward function names to statistics:
        {"reward_function_name": {"min": float, "max": float, "mean": float, "stddev": float}}
    """

    if config is None:
        raise ValueError("Config must be provided")

    print(f"Loading ground truth dataset using CustomDataset...")

    # Create CustomDataset - this is how the codebase loads the dataset
    try:
        custom_dataset = CustomDataset(
            cfg=config.dataset,
            split=config.dataset.validation.get("splits", ["test"]),
            ckpt_path=None,
        )
    except Exception as e:
        print(f"Error creating CustomDataset: {e}")
        raise

    total_scenes = len(custom_dataset)
    if num_scenes is None or num_scenes > total_scenes:
        num_scenes = total_scenes
    indices = list(range(num_scenes))
    print(f"Loading {num_scenes} scenes from dataset (total available: {total_scenes})")

    # Load scenes from dataset
    scenes_list = []
    for i in range(num_scenes):
        try:
            sample = custom_dataset[i]
            scene = sample["scenes"]
            scenes_list.append(scene)
        except Exception as e:
            print(f"Warning: Could not load scene {i}: {e}")
            continue

    if len(scenes_list) == 0:
        raise ValueError("No scenes could be loaded from the dataset")

    # Stack into batch tensor
    raw_results = torch.stack(scenes_list)
    print(f"Loaded scenes tensor shape: {raw_results.shape}")

    # Parse scenes
    parsed_scenes = parse_and_descale_scenes(raw_results)

    # Get room type and metadata from config
    room_type = config.algorithm.ddpo.dynamic_constraint_rewards.room_type
    all_rooms_info = json.load(
        open(
            os.path.join(
                config.dataset.data.path_to_dataset_files, "all_rooms_info.json"
            )
        )
    )
    idx_to_labels = all_rooms_info[room_type]["unique_values"]
    max_objects = all_rooms_info[room_type]["max_objects"]
    num_classes = all_rooms_info[room_type]["num_classes"]
    sdf_cache = SDFCache(config.dataset.sdf_cache_dir, split="test")
    reward_stats = {}
    floor_plan_args_list = [custom_dataset.get_floor_plan_args(idx) for idx in indices]
    # Stack each key across the batch for tensor conversion
    floor_plan_args = {
        key: [args[key] for args in floor_plan_args_list]
        for key in ["floor_plan_centroid", "floor_plan_vertices", "floor_plan_faces", "room_outer_box"]
    }
    # Compute rewards for each function
    for reward_name, reward_func in reward_functions.items():
        print(f"Computing rewards for: {reward_name}")
        rewards = reward_func(
            parsed_scenes,
            idx_to_labels=idx_to_labels,
            room_type=room_type,
            max_objects=max_objects,
            num_classes=num_classes,
            floor_polygons=[torch.tensor(custom_dataset.get_floor_polygon_points(idx), device=parsed_scenes["device"]) for idx in indices],
            indices=indices,
            is_val=True,
            sdf_cache=sdf_cache,
            floor_plan_args=floor_plan_args,
        )

        # Convert to numpy array
        if isinstance(rewards, torch.Tensor):
            rewards_array = rewards.cpu().numpy()
        else:
            rewards_array = np.array(rewards)

        # Compute statistics
        stats = {
            "min": float(np.min(rewards_array)),
            "max": float(np.max(rewards_array)),
            "mean": float(np.mean(rewards_array)),
            "median": float(np.median(rewards_array)),
            "stddev": float(np.std(rewards_array)),
            "percentile_1": float(np.percentile(rewards_array, 1)),
            "percentile_5": float(np.percentile(rewards_array, 5)),
            "percentile_25": float(np.percentile(rewards_array, 25)),
            "percentile_75": float(np.percentile(rewards_array, 75)),
            "percentile_95": float(np.percentile(rewards_array, 95)),
            "percentile_99": float(np.percentile(rewards_array, 99)),
            "num_scenes": len(rewards_array),
        }

        reward_stats[reward_name] = stats

        print(f"  Stats for {reward_name}:")
        print(f"    Min: {stats['min']:.4f}")
        print(f"    5th percentile: {stats['percentile_5']:.4f}")
        print(f"    25th percentile: {stats['percentile_25']:.4f}")
        print(f"    Median (50th): {stats['median']:.4f}")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    75th percentile: {stats['percentile_75']:.4f}")
        print(f"    95th percentile: {stats['percentile_95']:.4f}")
        print(f"    Max: {stats['max']:.4f}")
        print(f"    Stddev: {stats['stddev']:.4f}")

    return reward_stats
