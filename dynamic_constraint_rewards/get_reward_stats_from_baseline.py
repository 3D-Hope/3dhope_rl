"""
Script to compute reward statistics from baseline model samples.

This script samples scenes from a baseline model and computes statistics 
for multiple reward functions.
"""

import json
import os
import pickle
import subprocess

from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from omegaconf import DictConfig

from universal_constraint_rewards.commons import parse_and_descale_scenes
from steerable_scene_generation.datasets.custom_scene import CustomDataset
from universal_constraint_rewards.not_out_of_bound_reward import precompute_sdf_cache, SDFCache


def get_reward_stats_from_baseline(
    reward_functions: Dict[str, Callable],
    load: str = "rrudae6n",
    dataset: str = "custom_scene",
    config: DictConfig = None,
    dataset_processed_scene_data_path: str = "data/metadatas/custom_scene_metadata.json",
    dataset_max_num_objects_per_scene: int = 12,
    num_scenes: int = 1000,
    algorithm: str = "scene_diffuser_midiffusion",
    algorithm_trainer: str = "ddpm",
    experiment_find_unused_parameters: bool = True,
    algorithm_classifier_free_guidance_use: bool = False,
    algorithm_classifier_free_guidance_use_floor: bool = True,
    algorithm_classifier_free_guidance_weight: int = 1,
    algorithm_custom_loss: bool = True,
    algorithm_ema_use: bool = True,
    algorithm_noise_schedule_scheduler: str = "ddim",
    algorithm_noise_schedule_ddim_num_inference_timesteps: int = 150,
) -> Dict[str, Dict[str, float]]:
    """
    Sample scenes from baseline model and compute reward statistics.

    Args:
        reward_functions: Dict mapping reward function names to callable functions
                         Each function should take a scene dict and return a scalar reward [0, 1]
        load: Model checkpoint to load
        dataset: Dataset configuration name
        dataset_processed_scene_data_path: Path to scene metadata
        dataset_max_num_objects_per_scene: Maximum objects per scene
        num_scenes: Number of scenes to sample
        algorithm: Algorithm configuration name
        algorithm_trainer: Trainer type
        experiment_find_unused_parameters: Whether to find unused parameters
        algorithm_classifier_free_guidance_use: Use classifier-free guidance
        algorithm_classifier_free_guidance_weight: Guidance weight
        algorithm_custom_loss: Use custom loss
        algorithm_ema_use: Use EMA
        algorithm_noise_schedule_scheduler: Noise scheduler type
        algorithm_noise_schedule_ddim_num_inference_timesteps: Number of inference timesteps

    Returns:
        Dict mapping reward function names to statistics:
        {"reward_function_name": {"min": float, "max": float, "mean": float, "stddev": float}}
    """
    try:
        custom_dataset = CustomDataset(
            cfg=config.dataset,
            split=config.dataset.validation.get("splits", ["test"]),
            ckpt_path=None,
        )
    except Exception as e:
        print(f"Error creating CustomDataset: {e}")
        raise
    # Build command to run custom_sample_and_render.py
    cmd = [
        "python",
        "scripts/custom_sample_and_render.py",
        f"load={load}",
        f"dataset={dataset}",
        f"dataset.processed_scene_data_path={dataset_processed_scene_data_path}",
        f"dataset.max_num_objects_per_scene={dataset_max_num_objects_per_scene}",
        f"+num_scenes={num_scenes}",
        f"algorithm={algorithm}",
        f"algorithm.trainer={algorithm_trainer}",
        f"experiment.find_unused_parameters={experiment_find_unused_parameters}",
        f"algorithm.classifier_free_guidance.use={algorithm_classifier_free_guidance_use}",
        f"algorithm.classifier_free_guidance.use_floor={algorithm_classifier_free_guidance_use_floor}",
        f"algorithm.classifier_free_guidance.weight={algorithm_classifier_free_guidance_weight}",
        f"algorithm.custom.loss={str(algorithm_custom_loss).lower()}",
        f"algorithm.ema.use={algorithm_ema_use}",
        f"algorithm.noise_schedule.scheduler={algorithm_noise_schedule_scheduler}",
        f"algorithm.noise_schedule.ddim.num_inference_timesteps={algorithm_noise_schedule_ddim_num_inference_timesteps}",
    ]

    # Set PYTHONPATH environment variable
    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    print(f"Running sampling command...")
    print(f"Command: {' '.join(cmd)}")

    # Run the sampling script
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,  # Run from project root
    )

    print(f"Sampling script completed with return code {result}")
    if result.returncode != 0:
        print(f"Error running sampling script:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(
            f"Sampling script failed with return code {result.returncode}"
        )

    print(f"Sampling completed successfully")

    # Find the output pickle file
    # The script saves to outputs/<date>/<timestamp>/sampled_scenes_results.pkl
    outputs_dir = Path(__file__).parent.parent / "outputs"

    # Recursively find all pkl files and get the most recent one
    pkl_files = list(outputs_dir.glob("**/raw_sampled_scenes.pkl"))

    if not pkl_files:
        raise FileNotFoundError(
            f"Could not find sampled_scenes_results.pkl in outputs directory: {outputs_dir}"
        )

    # Sort by modification time and get the most recent
    pkl_path = max(pkl_files, key=lambda p: p.stat().st_mtime)

    print(f"Loading sampled scenes from: {pkl_path}")

    # Load the pickle file containing ThreedFrontResults
    with open(pkl_path, "rb") as f:
        raw_results = pickle.load(f)

    dataset_size = len(custom_dataset) 
    indices = [i % dataset_size for i in range(num_scenes)]
    # print(f"[Ashok] Raw results: {raw_results}")
    raw_results = torch.tensor(raw_results)
    parsed_scenes = parse_and_descale_scenes(raw_results)

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
        )

        # Compute statistics
        rewards_array = np.array(rewards)

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

