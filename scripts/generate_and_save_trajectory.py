"""
Script to generate denoising trajectories for multiple scenes in batch and save each trajectory
as a separate pickle file.


"""

import logging
import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
from threed_front.datasets import get_raw_dataset
from threed_front.evaluation import ThreedFrontResults
from tqdm import tqdm

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

    # Get batch size and base seed
    batch_size = cfg.get("batch_size", 16)
    base_seed = cfg.get("base_seed", None)
    if base_seed is None:
        base_seed = np.random.randint(0, 2**32 - 1)
    print(f"[INFO] Generating {batch_size} scenes with base seed: {base_seed}")

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    config = cfg.dataset

    # Check if load path is provided.
    if "load" not in cfg or cfg.load is None:
        raise ValueError("Please specify a checkpoint to load with 'load=...'")

    # Get scene index (which dataset sample to use for conditioning)
    scene_idx = cfg.get("scene_idx", 0)
    print(f"[INFO] Using dataset index: {scene_idx} for conditioning")

    # Get output directory for batch
    output_subdir = cfg.get("output_subdir", "trajectories")
    print(f"[INFO] Will save trajectories to subdirectory: {output_subdir}")

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

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

    # Load the checkpoint.
    load_id = cfg.load
    if is_run_id(load_id):
        # Download the checkpoint from wandb.
        run_path = f"{cfg.wandb.entity}/{cfg.wandb.project}/{load_id}"
        download_dir = output_dir / "checkpoints"
        version = cfg.get("checkpoint_version")
        if version is not None and isinstance(version, int):
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )
        else:
            checkpoint_path = download_latest_or_best_checkpoint(
                run_path=run_path,
                download_dir=download_dir,
                use_best=cfg.get("use_best", False),
            )
    else:
        # Use local path.
        checkpoint_path = Path(load_id)

    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    # Load datasets for postprocessing
    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"], config),
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        split=config["validation"].get("splits", ["test"]),
        max_length=config["max_num_objects_per_scene"],
        include_room_mask=config["network"].get("room_mask_condition", True),
    )

    # Create a CustomSceneDataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path),
    )

    print(f"[INFO] Dataset size: {len(custom_dataset)}")
    if scene_idx >= len(custom_dataset):
        raise ValueError(
            f"scene_idx={scene_idx} is out of bounds for dataset of size {len(custom_dataset)}"
        )

    # Build experiment
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    
    # Build algo and load checkpoint (same as exec_task does)
    algo = experiment._build_algo(ckpt_path=checkpoint_path)
    print(f"[DEBUG] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = algo.load_state_dict(state_dict, strict=False)
    
    # Load EMA if available
    if getattr(algo, "ema", None) and "ema_state_dict" in ckpt:
        algo.ema.load_state_dict(ckpt["ema_state_dict"])
        print(f"[DEBUG] Loaded EMA state dict")
    
    print(f"[DEBUG] Missing keys: {len(missing)}")
    print(f"[DEBUG] Unexpected keys: {len(unexpected)}")
    
    # Get the conditioning data for the specified scene index
    scene_data = custom_dataset[scene_idx]
    device = algo.device

    # Generate trajectory using the RL trainer's method
    algo.put_model_in_eval_mode()
    
    # Use EMA model if available (same as predict_step does)
    use_ema = cfg.algorithm.ema.use and cfg.algorithm.get("test", {}).get("use_ema", True)
    print(f"[INFO] Using EMA model: {use_ema}")
    
    # Create output subdirectory
    batch_output_dir = output_dir / output_subdir
    batch_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate trajectories for each scene in batch
    for batch_idx in range(batch_size):
        seed = base_seed + batch_idx
        print(f"\n[INFO] ========== Generating scene {batch_idx+1}/{batch_size} (seed={seed}) ==========")
    
        with torch.no_grad():
            # Manually generate trajectory (similar to generate_trajs_for_ddpo but without gradients)
            from diffusers import DDIMScheduler, DDPMScheduler
            
            room_type = getattr(cfg.dataset.data, "room_type", "bedroom")
            if isinstance(algo.noise_scheduler, DDIMScheduler):
                algo.noise_scheduler.set_timesteps(
                    cfg.algorithm.noise_schedule.ddim.num_inference_timesteps, device=device
                )

            trajectory = []
            
            # Set random seed RIGHT BEFORE sampling noise
            print(f"[INFO] Setting random seed {seed} before noise sampling")
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            
            # Sample initial noise
            num_objects_per_scene = (
                cfg.dataset.max_num_objects_per_scene
                + cfg.algorithm.num_additional_tokens_for_sampling
            )
            assert num_objects_per_scene == 12
            assert room_type == "bedroom"
            if room_type == "livingroom":
                xt = algo.sample_continuous_noise_prior(
                    (
                        1,  # Single scene
                        num_objects_per_scene,
                        cfg.algorithm.custom.num_classes
                        + cfg.algorithm.custom.translation_dim
                        + cfg.algorithm.custom.size_dim
                        + cfg.algorithm.custom.angle_dim
                        + cfg.algorithm.custom.objfeat_dim,
                    )
                ).to(device)
            elif room_type == "bedroom":
                xt = algo.sample_continuous_noise_prior(
                    (
                        1,  # Single scene
                        num_objects_per_scene,
                        algo.scene_vec_desc.get_object_vec_len(),
                    )
                ).to(device)
            else:
                raise ValueError(f"Unknown room type: {room_type}")
            
            trajectory.append(xt.clone()) # Append initial noise to trajectory
            
            # Create conditioning batch - IMPORTANT: Use the same structure as the dataloader
            # This should match what the model expects during training/inference
            data_batch = {
                "scenes": scene_data["scenes"].unsqueeze(0).to(device),  # Add batch dimension
                "idx": torch.tensor([scene_idx], device=device),
            }
            
            # Add other fields from scene_data if they exist
            for key in ["fpbpn", "text_cond", "text_cond_coarse"]:
                if key in scene_data:
                    val = scene_data[key]
                    if key == "fpbpn":
                        val = torch.tensor(val, device=device)
                    if isinstance(val, torch.Tensor):
                        data_batch[key] = val.unsqueeze(0).to(device)
                    else:
                        data_batch[key] = val
            
            # Denoising loop
            for t_idx, t in enumerate(
                tqdm(
                    algo.noise_scheduler.timesteps,
                    desc=f"Scene {batch_idx+1}/{batch_size}",
                )
            ):
                with torch.no_grad():
                    # Predict noise - use data_batch instead of cond_dict and use_ema=True
                    residual = algo.predict_noise(xt, t, cond_dict=data_batch, use_ema=use_ema)
                
                # Compute the updated sample.
                if isinstance(algo.noise_scheduler, DDIMScheduler):
                    scheduler_out = algo.noise_scheduler.step(
                        residual, t, xt, eta=cfg.algorithm.noise_schedule.ddim.eta
                    )
                else:
                    scheduler_out = algo.noise_scheduler.step(residual, t, xt)
                
                # Update the sample.
                xt = scheduler_out.prev_sample  # Shape (B, N, V)
                trajectory.append(xt.clone())
            
            # Stack trajectory: (T+1, 1, N, V) -> (T+1, N, V)
            trajectories = torch.stack(trajectory, dim=0).squeeze(1)

        # Convert to numpy
        trajectories_np = trajectories.detach().cpu().numpy()  # (T, N, V) 
        
        # Determine number of classes
        if cfg.dataset.data.room_type == "livingroom":
            n_classes = 25
        else:
            n_classes = 22
        
        # Postprocess each timestep as a separate scene
        print(f"[INFO] Postprocessing {trajectories_np.shape[0]} timesteps...")
        bbox_params_list = []
        
        for t in tqdm(range(trajectories_np.shape[0]), desc=f"Postprocessing scene {batch_idx+1}"):
            scene_at_t = trajectories_np[t]  # (N, V)
            
            class_labels, translations, sizes, angles, objfeats_32 = [], [], [], [], []
            
            for j in range(scene_at_t.shape[0]):
                class_label_idx = np.argmax(scene_at_t[j, 8:8+n_classes])
                if class_label_idx != n_classes - 1:  # ignore if empty token
                    ohe = np.zeros(n_classes - 1)
                    ohe[class_label_idx] = 1
                    class_labels.append(ohe)
                    translations.append(scene_at_t[j, :3])
                    sizes.append(scene_at_t[j,  3:6])
                    angles.append(scene_at_t[j,  6:8])
                    
                    try:
                        objfeats_32.append(
                            scene_at_t[j, n_classes + 8 : n_classes + 8 + 32]
                        )
                    except Exception:
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
        
        # Post-process and create layout list
        layout_list = []
        successful_timesteps = []
        
        for t, bbox_params_dict in enumerate(
            tqdm(bbox_params_list, desc=f"Post-processing scene {batch_idx+1}")
        ):
            try:
                boxes = encoded_dataset.post_process(bbox_params_dict)
                bbox_params = {k: v[0] for k, v in boxes.items()}
                layout_list.append(bbox_params)
                successful_timesteps.append(t)
            except Exception as e:
                print(f"[WARNING] Skipping timestep {t} due to post_process error: {e}")
                continue
        
        print(f"[INFO] Successfully postprocessed {len(layout_list)} out of {trajectories_np.shape[0]} timesteps")
        
        # Create indices list (all pointing to the same scene_idx)
        indices_list = [scene_idx] * len(layout_list)
        
        # Create ThreedFrontResults object
        threed_front_results = ThreedFrontResults(
            raw_train_dataset, raw_dataset, config, indices_list, layout_list
        )
        
        # Save to pickle - use the same format as custom_sample_and_render.py
        output_pkl_path = batch_output_dir / f"trajectory_seed{seed}.pkl"
        pickle.dump(threed_front_results, open(output_pkl_path, "wb"))
        
        print(f"[SUCCESS] Saved trajectory as {len(layout_list)} scenes to: {output_pkl_path}")
        print(f"[INFO] Timestep range: 0 to {len(successful_timesteps)-1}")
    
    print(f"\n[COMPLETE] Generated {batch_size} trajectories in: {batch_output_dir}")
    print(f"[INFO] You can now render each trajectory using:")
    print(f"       python ../ThreedFront/scripts/render_results.py --retrieve_by_size --no_texture --without_floor {batch_output_dir}/trajectory_seed<N>.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
