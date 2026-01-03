"""
Script for sampling from a trained model using the custom scene dataset.
"""

import logging
import os
import pickle
import sys

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults

from steerable_scene_generation.datasets.custom_scene import get_dataset_raw_and_encoded
from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
    CustomDataset,
    update_data_file_paths,
)
from steerable_scene_generation.experiments import build_experiment
from steerable_scene_generation.utils.ckpt_utils import (
    download_latest_or_best_checkpoint,
    download_version_checkpoint,
    is_run_id,
)
from steerable_scene_generation.utils.distributed_utils import is_rank_zero
from steerable_scene_generation.utils.logging import filter_drake_vtk_warning
from steerable_scene_generation.utils.omegaconf import register_resolvers

# Add logging filters.
filter_drake_vtk_warning()

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ[
    "HF_HOME"
] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
os.environ[
    "HF_DATASETS_CACHE"
] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"


@hydra.main(version_base=None, config_path="../configurations", config_name="config")
def main(cfg: DictConfig) -> None:
    if not is_rank_zero:
        raise ValueError(
            "This script must be run on the main process. "
            "Try export CUDA_VISIBLE_DEVICES=0."
        )

    # Set random seed.
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np

    np.random.seed(seed)

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset

    # Check if load path is provided.
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

    # Get configuration values with defaults.
    num_scenes = cfg.get("num_scenes", 1)
    print(f"[DEBUG] Number of scenes to sample: {num_scenes}")

    # Set predict mode.
    cfg.algorithm.predict.do_sample = True
    cfg.algorithm.predict.do_inference_time_search = False
    cfg.algorithm.predict.do_sample_scenes_with_k_closest_training_examples = False
    cfg.algorithm.predict.do_rearrange = False
    cfg.algorithm.predict.do_complete = False
    # print(f"[DEBUG] Predict config: {cfg}")
    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)
    print(f"[DEBUG] cfg_choice: {cfg_choice}")

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    logging.info(f"Outputs will be saved to: {output_dir}")

    # Initialize wandb.
    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.parent.name)
    load_id = cfg.load
    name = f"custom_sampling_{load_id}"
    wandb.init(
        name=name,
        dir=str(output_dir),
        config=OmegaConf.to_container(cfg),
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
    )

    # Load the checkpoint.
    if is_run_id(load_id):
        # Download the checkpoint from wandb.
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_dir = output_dir / "checkpoints"
        version = cfg["checkpoint_version"]
        print(f"[Ashok] checkpoint version from cfg: {version}")
        if version is not None and isinstance(version, int):
            print(f"[Ashok] downloading checkpoint version: {version}")
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )

        else:
            print(
                f"[Ashok] no checkpoint version specified, using_best {cfg.get('use_best', False)}"
            )
            checkpoint_path = download_latest_or_best_checkpoint(
                run_path=run_path,
                download_dir=download_dir,
                use_best=cfg.get("use_best", False),
            )
    else:
        # Use local path.
        checkpoint_path = Path(load_id)

    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"], config),
        # config["data"],
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    # Get Scaled dataset encoding (without data augmentation)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        # config["data"],
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    print(
        f"[Ashok] bounds sizes {encoded_dataset.bounds['sizes']}, translations {encoded_dataset.bounds['translations']}"
    )

    # print(f"[Ashok] type of dataset {type(encoded_dataset)}")
    print("[DEBUG] object_types:", len(raw_dataset.object_types))
    print("[DEBUG] class_labels:", len(raw_dataset.class_labels))
    # Create a CustomSceneDataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )

    # Save a ground truth sample from the dataset for comparison
    gt_sample_idx = 0  # Get the first sample
    gt_sample = custom_dataset[gt_sample_idx]
    # print(f"input {gt_sample}")
    # import sys; sys.exit(0)
    gt_sample_np = gt_sample["scenes"].detach().cpu().numpy()
    gt_text_path = output_dir / "ground_truth_scene.txt"
    with open(gt_text_path, "w") as f:
        f.write(f"Ground truth scene shape: {gt_sample_np.shape}\n\n")
        f.write(f"Sample index: {gt_sample_idx}\n\n")
        f.write(np.array2string(gt_sample_np, threshold=np.inf, precision=6))

    logging.info(f"Saved ground truth scene to {gt_text_path}")

    # Limit dataset to num_scenes samples
    dataset_size = len(custom_dataset)
    num_scenes_to_sample = num_scenes  # Always use requested num_scenes

    # Create indices with resampling if needed
    if num_scenes_to_sample <= dataset_size:
        # Use first num_scenes samples without resampling
        indices = list(range(num_scenes_to_sample))
    else:
        # Need to resample: repeat the dataset multiple times
        print(
            f"[INFO] Requested {num_scenes_to_sample} scenes but dataset only has {dataset_size} scenes."
        )
        print(
            f"[INFO] Will resample with replacement to generate {num_scenes_to_sample} scenes."
        )
        indices = [i % dataset_size for i in range(num_scenes_to_sample)]

    # Create subset of dataset with the indices (may include duplicates)
    from torch.utils.data import Subset

    limited_dataset = Subset(custom_dataset, indices)

    # Store the actual dataset indices for each sample (needed for conditions and floor rendering)
    sampled_dataset_indices = (
        indices.copy()
    )  # This tracks which dataset index was used for each sample

    print(f"[DEBUG] Full dataset size: {dataset_size}")
    print(f"[DEBUG] Sampling {num_scenes_to_sample} scenes")
    print(f"[DEBUG] Number of unique scenes: {min(dataset_size, num_scenes_to_sample)}")
    print(f"[DEBUG] Sample indices (first 10): {sampled_dataset_indices[:10]}")

    # Use batch size from config, not num_scenes
    batch_size = cfg.experiment.get("test", {}).get(
        "batch_size", cfg.experiment.validation.batch_size
    )
    print(f"[DEBUG] Using batch size: {batch_size}")

    # Create a dataloader for the limited dataset
    dataloader = torch.utils.data.DataLoader(
        limited_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
        pin_memory=cfg.experiment.test.pin_memory,
    )

    print(f"[DEBUG] Created limited dataset with size: {len(limited_dataset)}")

    # Build experiment with custom dataset
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)

    try:
        print("[DEBUG] Starting to sample scenes...")
        # Sample scenes from the model
        sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
        # diffuser = experiment.algo
        # svd = diffuser.scene_vec_desc
        # print("[DEBUG] object vec len:", svd.get_object_vec_len())
        # print("[DEBUG] class (model_path) vec len:", svd.get_model_path_vec_len())
        # print("[DEBUG] translation vec len:", svd.get_translation_vec_len())
        # print("[DEBUG] rotation vec len:", svd.get_rotation_vec_len())
        # import sys; sys.exit(0)

        # Use the actual dataset indices that were sampled (handles resampling with replacement)
        # sampled_dataset_indices contains the original dataset index for each sampled scene
        sampled_indices = sampled_dataset_indices

        print(f"[DEBUG] Number of sampled scenes: {len(sampled_indices)}")
        print(
            f"[DEBUG] Sampled indices length matches batches: {len(sampled_indices) == sum(len(batch) for batch in sampled_scene_batches)}"
        )

        sampled_scenes = torch.cat(sampled_scene_batches, dim=0)

        print(f"[DEBUG] Sampled scenes shape: {sampled_scenes.shape}")
        print(f"[DEBUG] Sampled indices count: {len(sampled_indices)}")

        # Verify the counts match
        assert (
            len(sampled_indices) == sampled_scenes.shape[0]
        ), f"Mismatch: {len(sampled_indices)} indices vs {sampled_scenes.shape[0]} scenes"

        with open(output_dir / "raw_sampled_scenes.pkl", "wb") as f:
            pickle.dump(sampled_scenes, f)

        # Remove samples that contain any NaN and keep indices in sync
        mask = ~torch.any(torch.isnan(sampled_scenes), dim=(1, 2))
        print(f"[DEBUG] Number of scenes with NaN values: {torch.sum(~mask)}")
        # Filter scenes
        sampled_scenes = sampled_scenes[mask]
        # Filter corresponding dataset indices so they align with the kept scenes
        mask_np = mask.detach().cpu().numpy().astype(bool)
        sampled_indices = [idx for idx, keep in zip(sampled_indices, mask_np) if keep]

        print(
            f"Remaining samples: {sampled_scenes.shape[0]} out of {num_scenes_to_sample}"
        )

        sampled_scenes_np = sampled_scenes.detach().cpu().numpy()  # b, 12, 30
        print(f"[Ashok] sampled scene {sampled_scenes_np[0]}")
        bbox_params_list = []
        if cfg.dataset.data.room_type == "livingroom":
            n_classes = 25
        else:
            n_classes = 22
        path_to_results = output_dir / "sampled_scenes_results.pkl"
        for i in range(sampled_scenes_np.shape[0]):
            class_labels, translations, sizes, angles, objfeats_32 = [], [], [], [], []
            for j in range(sampled_scenes_np.shape[1]):
                class_label_idx = np.argmax(sampled_scenes_np[i, j, 8 : 8 + n_classes])
                if class_label_idx != n_classes - 1:  # ignore if empty token
                    ohe = np.zeros(n_classes - 1)
                    ohe[class_label_idx] = 1
                    class_labels.append(ohe)
                    translations.append(sampled_scenes_np[i, j, 0:3])
                    sizes.append(sampled_scenes_np[i, j, 3:6])
                    angles.append(sampled_scenes_np[i, j, 6:8])
                    try:
                        objfeats_32.append(
                            sampled_scenes_np[i, j, 8 + n_classes : n_classes + 8 + 32]
                        )

                    except Exception as e:
                        objfeats_32 = None

            bbox_params_list.append(
                {
                    "class_labels": np.array(class_labels)[None, :],
                    "translations": np.array(translations)[None, :],
                    "sizes": np.array(sizes)[None, :],
                    "angles": np.array(angles)[None, :],
                    "objfeats_32": np.array(objfeats_32)[None, :]
                    if objfeats_32 is not None
                    else None,
                }
            )
        # print(f"[Ashok] shape of objfeats_32: {np.array(objfeats_32)[None, :].shape}")
        # # print("bbox param list", bbox_params_list)
        # print(f"[Ashok] type of dataset {type(encoded_dataset)}")

        # Only collect successful indices and layouts
        layout_list = []
        successful_indices = []
        for idx, bbox_params_dict in enumerate(bbox_params_list):
            try:
                boxes = encoded_dataset.post_process(bbox_params_dict)
                bbox_params = {k: v[0] for k, v in boxes.items()}
                layout_list.append(bbox_params)
                successful_indices.append(sampled_indices[idx])
            except Exception as e:
                print(f"[WARNING] Skipping scene {idx} due to post_process error: {e}")
                continue

        # print("postprocessed output: ", layout_list[0])
        # layout_list [{"class_labels":[], "translations":[1,2,3], "sizes": [1,2,3,], "angles": [1]}, ...]
        threed_front_results = ThreedFrontResults(
            raw_train_dataset, raw_dataset, config, successful_indices, layout_list
        )
        pickle.dump(threed_front_results, open(path_to_results, "wb"))
        print("Saved result to:", path_to_results)

        return path_to_results
        kl_divergence = threed_front_results.kl_divergence()
        print("object category kl divergence:", kl_divergence)

        # Calculate and save object statistics for both generated and ground truth scenes
        import json

        from collections import Counter, defaultdict

        import numpy as np

        # Get object category mapping from model paths to readable names
        def extract_object_name_from_path(path):
            """Extract a readable object name from the model path"""
            if path is None:
                return "empty"
            # Extract the last part of the path (after the last '/')
            parts = path.split("/")
            # Get the filename without extension
            obj_name = (
                parts[-2]
                if parts[-1] == "model.sdf" or parts[-1] == "model_simplified.sdf"
                else parts[-1].split(".")[0]
            )
            # Clean up the name to make it readable
            obj_name = obj_name.replace("_", " ").lower()
            return obj_name

        # Collect statistics for generated scenes
        gen_stats = defaultdict(list)
        gen_category_counts = Counter()

        for i, scene_params in enumerate(layout_list):
            class_labels = scene_params["class_labels"]
            object_count = len(class_labels)
            gen_stats["object_counts"].append(object_count)

            # Get object category indices safely
            try:
                for obj_idx in range(len(class_labels)):
                    try:
                        # Handle both numpy arrays and lists
                        if isinstance(class_labels[obj_idx], (np.ndarray, list)):
                            cat_idx = np.argmax(class_labels[obj_idx])
                        else:
                            # Might already be a category index
                            cat_idx = class_labels[obj_idx]

                        # Get class label mapping
                        class_label_map = None
                        if hasattr(raw_dataset, "class_labels"):
                            class_label_map = raw_dataset.class_labels

                        if class_label_map is not None and cat_idx < len(
                            class_label_map
                        ):
                            obj_name = extract_object_name_from_path(
                                class_label_map[cat_idx]
                            )
                        else:
                            # If we can't find the mapping, use a generic name with index
                            obj_name = f"object_type_{cat_idx}"

                        gen_category_counts[obj_name] += 1
                    except Exception as e:
                        print(f"Error processing generated object {obj_idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing generated objects: {str(e)}")
                continue

        # Calculate statistics for generated scenes - convert numpy types to Python standard types
        gen_stats["total_scenes"] = int(len(layout_list))
        gen_stats["avg_objects_per_scene"] = float(np.mean(gen_stats["object_counts"]))
        gen_stats["std_objects_per_scene"] = float(np.std(gen_stats["object_counts"]))
        gen_stats["min_objects"] = int(np.min(gen_stats["object_counts"]))
        gen_stats["max_objects"] = int(np.max(gen_stats["object_counts"]))
        # Convert object counts to regular Python types before JSON serialization
        gen_stats["object_counts"] = [int(x) for x in gen_stats["object_counts"]]
        # Convert category counts from numpy to standard Python types
        gen_category_counts_py = {k: int(v) for k, v in gen_category_counts.items()}
        gen_stats["category_frequencies"] = {
            k: float(v / sum(gen_category_counts_py.values()))
            for k, v in sorted(
                gen_category_counts_py.items(), key=lambda x: x[1], reverse=True
            )
        }
        gen_stats["category_counts"] = {
            k: int(v)
            for k, v in sorted(
                gen_category_counts_py.items(), key=lambda x: x[1], reverse=True
            )
        }

        # Collect statistics for ground truth scenes (from raw_dataset)
        gt_stats = defaultdict(list)
        gt_category_counts = Counter()

        # Limit to a reasonable number of GT samples for performance
        for idx in range(min(100, len(raw_dataset))):
            try:
                sample = raw_dataset[idx]
                # Handle different attribute access patterns
                if hasattr(sample, "class_labels"):
                    class_labels = sample.class_labels
                elif isinstance(sample, dict) and "class_labels" in sample:
                    class_labels = sample["class_labels"]
                else:
                    continue
                object_count = len(class_labels)
                gt_stats["object_counts"].append(object_count)
            except Exception as e:
                print(f"Error processing GT sample {idx}: {str(e)}")
                continue

            # Safely get class labels
            try:
                for obj_idx in range(len(class_labels)):
                    try:
                        # Handle both numpy arrays and lists
                        if isinstance(class_labels[obj_idx], (np.ndarray, list)):
                            cat_idx = np.argmax(class_labels[obj_idx])
                        else:
                            # Might already be a category index
                            cat_idx = class_labels[obj_idx]

                        # Get class label mapping
                        class_label_map = None
                        if hasattr(raw_dataset, "class_labels"):
                            class_label_map = raw_dataset.class_labels

                        if class_label_map is not None and cat_idx < len(
                            class_label_map
                        ):
                            obj_name = extract_object_name_from_path(
                                class_label_map[cat_idx]
                            )
                        else:
                            # If we can't find the mapping, use a generic name with index
                            obj_name = f"object_type_{cat_idx}"

                        gt_category_counts[obj_name] += 1
                    except Exception as e:
                        print(f"Error processing GT object {obj_idx}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error processing GT objects: {str(e)}")
                continue

        # Calculate statistics for ground truth scenes - convert numpy types to Python standard types
        gt_stats["total_scenes"] = int(len(gt_stats["object_counts"]))
        gt_stats["avg_objects_per_scene"] = float(np.mean(gt_stats["object_counts"]))
        gt_stats["std_objects_per_scene"] = float(np.std(gt_stats["object_counts"]))
        gt_stats["min_objects"] = int(np.min(gt_stats["object_counts"]))
        gt_stats["max_objects"] = int(np.max(gt_stats["object_counts"]))
        # Convert object counts to regular Python types before JSON serialization
        gt_stats["object_counts"] = [int(x) for x in gt_stats["object_counts"]]
        # Convert category counts from numpy to standard Python types
        gt_category_counts_py = {k: int(v) for k, v in gt_category_counts.items()}
        gt_stats["category_frequencies"] = {
            k: float(v / sum(gt_category_counts_py.values()))
            for k, v in sorted(
                gt_category_counts_py.items(), key=lambda x: x[1], reverse=True
            )
        }
        gt_stats["category_counts"] = {
            k: int(v)
            for k, v in sorted(
                gt_category_counts_py.items(), key=lambda x: x[1], reverse=True
            )
        }

        # Combine both stats
        all_stats = {"generated": gen_stats, "ground_truth": gt_stats}

        # Save statistics to JSON file
        stats_path = output_dir / "scene_statistics.json"
        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)

        # Print summary statistics
        print("\n===== SCENE STATISTICS =====")
        print(
            f"Generated scenes: {gen_stats['total_scenes']} scenes, "
            f"{gen_stats['avg_objects_per_scene']:.2f}±{gen_stats['std_objects_per_scene']:.2f} objects per scene"
        )
        print(
            f"Ground truth: {gt_stats['total_scenes']} scenes, "
            f"{gt_stats['avg_objects_per_scene']:.2f}±{gt_stats['std_objects_per_scene']:.2f} objects per scene"
        )

        print("\nTop 5 object categories in generated scenes:")
        for i, (cat, freq) in enumerate(
            list(gen_stats["category_frequencies"].items())[:5]
        ):
            print(
                f"  {i+1}. {cat}: {freq:.1%} ({gen_stats['category_counts'][cat]} instances)"
            )

        print("\nTop 5 object categories in ground truth scenes:")
        for i, (cat, freq) in enumerate(
            list(gt_stats["category_frequencies"].items())[:5]
        ):
            print(
                f"  {i+1}. {cat}: {freq:.1%} ({gt_stats['category_counts'][cat]} instances)"
            )

        print(f"\nStatistics saved to: {stats_path}")

    ###---
    # # Log to wandb and save locally
    # pickle_path = output_dir / "sampled_scenes.pkl"
    # with open(pickle_path, "wb") as f:
    #     pickle.dump(scene_dict, f)

    # # Also save as text file for easy inspection
    # text_path = output_dir / "sampled_scenes.txt"
    # with open(text_path, "w") as f:
    #     f.write(f"Sampled scenes shape: {sampled_scenes_np.shape}\n\n")
    #     f.write(np.array2string(sampled_scenes_np, threshold=np.inf, precision=6))

    # logging.info(f"Saved sampled scenes to {pickle_path} and {text_path}")

    # # Log to wandb
    # wandb.save(str(pickle_path))
    # wandb.save(str(text_path))

    except Exception as e:
        logging.error(f"Error during sampling: {str(e)}")
        # Still try to save any partial results
        try:
            if "sampled_scenes" in locals():
                sampled_scenes_np = sampled_scenes.detach().cpu().numpy()
                text_path = output_dir / "partial_sampled_scenes.txt"
                with open(text_path, "w") as f:
                    f.write(
                        f"Partial sampled scenes shape: {sampled_scenes_np.shape}\n\n"
                    )
                    f.write(
                        np.array2string(
                            sampled_scenes_np, threshold=np.inf, precision=6
                        )
                    )
                logging.info(f"Saved partial sampled scenes to {text_path}")
                wandb.save(str(text_path))
        except:
            pass

        raise e

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
