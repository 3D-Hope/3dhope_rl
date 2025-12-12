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
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
from steerable_scene_generation.algorithms.scene_diffusion.ddpo_helpers import (
    ddim_step_with_logprob,
    ddpm_step_with_logprob,
)
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

    # Set random seed
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

    # Get configuration values
    num_scenes = cfg.get("num_scenes", 1)
    print(f"[INFO] Number of scenes to generate trajectories for: {num_scenes}")

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
    
    # Create indices with resampling if needed (similar to custom_sample_and_render.py)
    dataset_size = len(custom_dataset)
    num_scenes_to_sample = num_scenes
    
    if num_scenes_to_sample <= dataset_size:
        indices = list(range(num_scenes_to_sample))
    else:
        print(f"[INFO] Requested {num_scenes_to_sample} scenes but dataset only has {dataset_size} scenes.")
        print(f"[INFO] Will resample with replacement to generate {num_scenes_to_sample} scenes.")
        indices = [i % dataset_size for i in range(num_scenes_to_sample)]
    
    # Create subset of dataset with the indices
    from torch.utils.data import Subset
    limited_dataset = Subset(custom_dataset, indices)
    
    # Store the actual dataset indices for each sample
    sampled_dataset_indices = indices.copy()
    
    print(f"[DEBUG] Full dataset size: {dataset_size}")
    print(f"[DEBUG] Sampling {num_scenes_to_sample} scenes")
    print(f"[DEBUG] Sample indices (first 10): {sampled_dataset_indices[:10]}")
    
    # Use batch size from config
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
    
    device = algo.device

    print(f"[INFO] Generating trajectories on device: {device}")
    # Generate trajectory using the RL trainer's method
    algo.put_model_in_eval_mode()
    
    # Use EMA model if available (same as predict_step does)
    use_ema = cfg.algorithm.ema.use and cfg.algorithm.get("test", {}).get("use_ema", True)
    print(f"[INFO] Using EMA model: {use_ema}")
    
    # Create output subdirectory
    batch_output_dir = output_dir / output_subdir
    batch_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Import scheduler types
    from diffusers import DDIMScheduler, DDPMScheduler
    
    room_type = getattr(cfg.dataset.data, "room_type", "bedroom")
    if isinstance(algo.noise_scheduler, DDIMScheduler):
        algo.noise_scheduler.set_timesteps(
            cfg.algorithm.noise_schedule.ddim.num_inference_timesteps, device=device
        )
    
    num_objects_per_scene = (
        cfg.dataset.max_num_objects_per_scene
        + cfg.algorithm.num_additional_tokens_for_sampling
    )
    
    # Determine number of classes
    if cfg.dataset.data.room_type == "livingroom":
        n_classes = 25
    else:
        n_classes = 22
    
    # Process batches from dataloader
    global_scene_idx = 0
    
    # Dictionary to store loss trajectories for plotting
    # Format: {scene_idx: {"timesteps": [], "pos_loss": [], "size_loss": [], ...}}
    loss_trajectories = {}
    
    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
        current_batch_size = batch_data["scenes"].shape[0]
        print(f"\n[INFO] ========== Processing batch {batch_idx+1}/{len(dataloader)} ({current_batch_size} scenes) ==========")
        
        with torch.no_grad():
            # Sample initial noise for entire batch
            if room_type == "bedroom":
                xt = algo.sample_continuous_noise_prior(
                    (
                        current_batch_size,
                        num_objects_per_scene,
                        algo.scene_vec_desc.get_object_vec_len(),
                    )
                ).to(device)
            elif room_type == "livingroom":
                xt = algo.sample_continuous_noise_prior(
                    (
                        current_batch_size,
                        num_objects_per_scene,
                        cfg.algorithm.custom.num_classes
                        + cfg.algorithm.custom.translation_dim
                        + cfg.algorithm.custom.size_dim
                        + cfg.algorithm.custom.angle_dim
                        + cfg.algorithm.custom.objfeat_dim,
                    )
                ).to(device)
            else:
                raise ValueError(f"Unknown room type: {room_type}")
            
            # Store trajectory for entire batch: shape will be (T, B, N, V)
            trajectories_batch = [xt.clone()]
            
            # Prepare conditioning batch
            data_batch = {
                "scenes": batch_data["scenes"].to(device),
                "idx": batch_data.get("idx", torch.arange(current_batch_size)).to(device),
            }
            
            for key in ["fpbpn", "text_cond", "text_cond_coarse"]:
                if key in batch_data:
                    val = batch_data[key]
                    if key == "fpbpn" and not isinstance(val, torch.Tensor):
                        val = torch.tensor(val, device=device)
                    if isinstance(val, torch.Tensor):
                        data_batch[key] = val.to(device)
                    else:
                        data_batch[key] = val
            
            # Denoising loop for entire batch
            for t_idx, t in enumerate(
                tqdm(algo.noise_scheduler.timesteps, desc=f"Denoising batch {batch_idx+1}")
            ):
                with torch.no_grad():
                    residual = algo.predict_noise(xt, t, cond_dict=data_batch, use_ema=use_ema)
                
                # if isinstance(algo.noise_scheduler, DDIMScheduler):
                #     scheduler_out = algo.noise_scheduler.step(
                #         residual, t, xt, eta=cfg.algorithm.noise_schedule.ddim.eta
                #     )
                # else:
                #     scheduler_out = algo.noise_scheduler.step(residual, t, xt)
                
                if isinstance(algo.noise_scheduler, DDPMScheduler):
                    xt_next, log_prop = ddpm_step_with_logprob(
                        scheduler=algo.noise_scheduler,
                        model_output=residual,
                        timestep=t,
                        sample=xt,
                        # mask=inpainting_masks if use_inpaint else None,
                    )
                else:  # DDIMScheduler
                    xt_next, log_prop = ddim_step_with_logprob(
                        scheduler=algo.noise_scheduler,
                        model_output=residual,
                        timestep=t,
                        sample=xt,
                        eta=algo.cfg.noise_schedule.ddim.eta,
                        # mask=inpainting_masks if use_inpaint else None,
                    )
                xt = xt_next
                trajectories_batch.append(xt.clone())
            
            # Stack: (T+1, B, N, V)
            trajectories_batch = torch.stack(trajectories_batch, dim=0)
        
        # Compute loss trajectories for each scene in batch
        # Get ground truth scenes from batch_data
        gt_scenes = batch_data["scenes"].to(device)  # (B, N, V)
        
        # Define indices for loss components (same as in custom_loss_function)
        pos_indices = list(range(0, 3))
        size_indices = list(range(len(pos_indices), len(pos_indices) + 3))
        rot_indices = list(range(len(pos_indices) + len(size_indices), len(pos_indices) + len(size_indices) + 2))
        class_indices = list(range(len(pos_indices) + len(size_indices) + len(rot_indices), 
                                   len(pos_indices) + len(size_indices) + len(rot_indices) + n_classes))
        
        # Calculate losses at each timestep for entire batch
        for t_idx in range(trajectories_batch.shape[0]):
            pred_at_t = trajectories_batch[t_idx]  # (B, N, V)
            
            # Extract components
            pred_pos = pred_at_t[..., pos_indices]
            pred_size = pred_at_t[..., size_indices]
            pred_rot = pred_at_t[..., rot_indices]
            pred_class = pred_at_t[..., class_indices]
            
            gt_pos = gt_scenes[..., pos_indices]
            gt_size = gt_scenes[..., size_indices]
            gt_rot = gt_scenes[..., rot_indices]
            gt_class = gt_scenes[..., class_indices]
            
            # Compute losses per scene (B,)
            pos_loss_batch = F.mse_loss(pred_pos, gt_pos, reduction='none').mean(dim=[1, 2])
            size_loss_batch = F.mse_loss(pred_size, gt_size, reduction='none').mean(dim=[1, 2])
            rot_loss_batch = F.mse_loss(pred_rot, gt_rot, reduction='none').mean(dim=[1, 2])
            class_loss_batch = F.mse_loss(pred_class, gt_class, reduction='none').mean(dim=[1, 2])
            
            # Store losses for each scene in the batch
            for scene_in_batch_idx in range(current_batch_size):
                scene_id = global_scene_idx + scene_in_batch_idx
                if scene_id not in loss_trajectories:
                    loss_trajectories[scene_id] = {
                        "timesteps": [],
                        "pos_loss": [],
                        "size_loss": [],
                        "rot_loss": [],
                        "class_loss": [],
                        "total_loss": []
                    }
                
                loss_trajectories[scene_id]["timesteps"].append(t_idx)
                loss_trajectories[scene_id]["pos_loss"].append(pos_loss_batch[scene_in_batch_idx].item())
                loss_trajectories[scene_id]["size_loss"].append(size_loss_batch[scene_in_batch_idx].item())
                loss_trajectories[scene_id]["rot_loss"].append(rot_loss_batch[scene_in_batch_idx].item())
                loss_trajectories[scene_id]["class_loss"].append(class_loss_batch[scene_in_batch_idx].item())
                
                total = (pos_loss_batch[scene_in_batch_idx] + size_loss_batch[scene_in_batch_idx] + 
                        rot_loss_batch[scene_in_batch_idx] + class_loss_batch[scene_in_batch_idx]).item()
                loss_trajectories[scene_id]["total_loss"].append(total)
        
        # Process each scene in the batch separately
        for scene_in_batch_idx in range(current_batch_size):
            print(f"\n[INFO] Processing scene {global_scene_idx+1}/{num_scenes_to_sample}")
            
            # Extract trajectory for this scene: (T+1, N, V)
            trajectory_single = trajectories_batch[:, scene_in_batch_idx, :, :].detach().cpu().numpy()
            
            # Postprocess each timestep
            bbox_params_list = []
            for t in range(trajectory_single.shape[0]):
                scene_at_t = trajectory_single[t]  # (N, V)
                
                class_labels, translations, sizes, angles, objfeats_32 = [], [], [], [], []
                
                for j in range(scene_at_t.shape[0]):
                    class_label_idx = np.argmax(scene_at_t[j, 8:8+n_classes])
                    if class_label_idx != n_classes - 1:
                        ohe = np.zeros(n_classes - 1)
                        ohe[class_label_idx] = 1
                        class_labels.append(ohe)
                        translations.append(scene_at_t[j, :3])
                        sizes.append(scene_at_t[j, 3:6])
                        angles.append(scene_at_t[j, 6:8])
                        
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
            
            # Post-process layouts
            layout_list = []
            successful_timesteps = []
            
            for t, bbox_params_dict in enumerate(bbox_params_list):
                try:
                    boxes = encoded_dataset.post_process(bbox_params_dict)
                    bbox_params = {k: v[0] for k, v in boxes.items()}
                    layout_list.append(bbox_params)
                    successful_timesteps.append(t)
                except Exception as e:
                    print(f"[WARNING] Scene {global_scene_idx}, timestep {t}: {e}")
                    continue
            
            # Get the original dataset index for this scene
            original_idx = sampled_dataset_indices[global_scene_idx]
            indices_list = [original_idx] * len(layout_list)
            
            # Create ThreedFrontResults
            threed_front_results = ThreedFrontResults(
                raw_train_dataset, raw_dataset, config, indices_list, layout_list
            )
            
            # Save trajectory
            output_pkl_path = batch_output_dir / f"trajectory_scene{global_scene_idx}.pkl"
            pickle.dump(threed_front_results, open(output_pkl_path, "wb"))
            
            print(f"[SUCCESS] Saved trajectory with {len(layout_list)} timesteps to: {output_pkl_path}")
            
            global_scene_idx += 1 
    
    print(f"\n[COMPLETE] Generated {global_scene_idx} trajectories in: {batch_output_dir}")
    
    # Plot loss evolution
    print(f"\n[INFO] Plotting loss evolution...")
    plot_output_dir = batch_output_dir / "loss_plots"
    plot_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot 1: Total loss for all scenes
    plt.figure(figsize=(12, 8))
    for scene_id, losses in loss_trajectories.items():
        plt.plot(losses["timesteps"], losses["total_loss"], label=f"Scene {scene_id}", alpha=0.7)
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Total Loss (MSE)", fontsize=12)
    plt.title("Denoising Loss Evolution - Total Loss", fontsize=14)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    total_loss_path = plot_output_dir / "total_loss_evolution.png"
    plt.savefig(total_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved total loss plot to: {total_loss_path}")
    
    # Plot 2: Component losses (averaged across all scenes)
    plt.figure(figsize=(12, 8))
    
    # Average losses across all scenes
    max_timesteps = max(len(losses["timesteps"]) for losses in loss_trajectories.values()) # 151
    # print("[DEBUG] Max timesteps across scenes:", max_timesteps)
    avg_pos_loss = np.zeros(max_timesteps)
    avg_size_loss = np.zeros(max_timesteps)
    avg_rot_loss = np.zeros(max_timesteps)
    avg_class_loss = np.zeros(max_timesteps)
    avg_total_loss = np.zeros(max_timesteps)
    counts = np.zeros(max_timesteps)
    
    for scene_id, losses in loss_trajectories.items():
        for i, t in enumerate(losses["timesteps"]):
            avg_pos_loss[t] += losses["pos_loss"][i]
            avg_size_loss[t] += losses["size_loss"][i]
            avg_rot_loss[t] += losses["rot_loss"][i]
            avg_class_loss[t] += losses["class_loss"][i]
            avg_total_loss[t] += losses["total_loss"][i]
            counts[t] += 1
    # print(f"[DEBUG] Counts per timestep:", counts)
    # Normalize by counts
    avg_pos_loss /= np.maximum(counts, 1)
    avg_size_loss /= np.maximum(counts, 1)
    avg_rot_loss /= np.maximum(counts, 1)
    avg_class_loss /= np.maximum(counts, 1)
    avg_total_loss /= np.maximum(counts, 1)
    
    timesteps = list(range(max_timesteps))
    plt.plot(timesteps, avg_pos_loss, label="Position Loss", linewidth=2)
    plt.plot(timesteps, avg_size_loss, label="Size Loss", linewidth=2)
    plt.plot(timesteps, avg_rot_loss, label="Rotation Loss", linewidth=2)
    plt.plot(timesteps, avg_class_loss, label="Class Loss", linewidth=2)
    # plt.plot(timesteps, avg_total_loss, label="Total Loss", linewidth=2.5, linestyle='--', color='black')
    
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Loss (MSE)", fontsize=12)
    plt.title("Denoising Loss Evolution - Component Losses Averaged(over 1000 scenes)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    component_loss_path = plot_output_dir / "component_loss_evolution.png"
    plt.savefig(component_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved component loss plot to: {component_loss_path}")
    
    # Plot 3: Averaged total loss across all scenes
    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, avg_total_loss, label="Average(over 1000 scenes) Total Loss", linewidth=2.5, color='black')
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Total Loss (MSE)", fontsize=12)
    plt.title("Denoising Loss Evolution - Average Total Loss", fontsize=14)
    # plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    avg_total_loss_path = plot_output_dir / "average_total_loss_evolution.png"
    plt.savefig(avg_total_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved average total loss plot to: {avg_total_loss_path}")
    
    # Plot 4: Individual component losses per scene (separate subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    component_names = ["pos_loss", "size_loss", "rot_loss", "class_loss"]
    component_titles = ["Position Loss", "Size Loss", "Rotation Loss", "Class Loss"]
    
    for idx, (comp_name, comp_title) in enumerate(zip(component_names, component_titles)):
        ax = axes[idx // 2, idx % 2]
        for scene_id, losses in loss_trajectories.items():
            ax.plot(losses["timesteps"], losses[comp_name], label=f"Scene {scene_id}", alpha=0.7)
        ax.set_xlabel("Timestep", fontsize=10)
        ax.set_ylabel("Loss (MSE)", fontsize=10)
        ax.set_title(comp_title, fontsize=12)
        # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    individual_loss_path = plot_output_dir / "individual_component_losses.png"
    plt.savefig(individual_loss_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved individual component loss plot to: {individual_loss_path}")
    
    # Save loss data as pickle for later analysis
    loss_data_path = plot_output_dir / "loss_trajectories.pkl"
    with open(loss_data_path, "wb") as f:
        pickle.dump(loss_trajectories, f)
    print(f"[SUCCESS] Saved loss trajectory data to: {loss_data_path}")
    
    print(f"[INFO] You can now render each trajectory using:")
    print(f"       python ../ThreedFront/scripts/render_results.py --retrieve_by_size --no_texture --without_floor {batch_output_dir}/trajectory_scene<N>.pkl")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
