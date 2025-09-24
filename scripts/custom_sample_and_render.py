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

from steerable_scene_generation.algorithms.scene_diffusion.scene_diffuser_base import (
    SceneDiffuserBase,
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

from steerable_scene_generation.datasets.custom_scene.custom_scene_final import (
    CustomDataset
)

from threed_front.evaluation import ThreedFrontResults
from threed_front.datasets import get_raw_dataset
from steerable_scene_generation.datasets.custom_scene import get_dataset_raw_and_encoded
from steerable_scene_generation.datasets.custom_scene.custom_scene_final import update_data_file_paths
# Add logging filters.
filter_drake_vtk_warning()

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/.cache/huggingface/datasets"

# config = {'data': {'dataset_directory': '/mnt/sv-share/MiData/bedroom', 'dataset_type': 'cached_threedfront', 'encoding_type': 'cached_diffusion_cosin_angle_wocm_no_prm_eval', 'annotation_file': '/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/MiDiffusion/../ThreedFront/dataset_files/bedroom_threed_front_splits_original.csv', 'augmentations': ['fixed_rotations'], 'train_stats': 'dataset_stats.txt', 'room_layout_size': '64,64', 'room_type': 'bedroom'}, 'network': {'type': 'diffusion_scene_layout_mixed', 'sample_num_points': 12, 'max_cuboids': 19, 'angle_dim': 2, 'room_mask_condition': True, 'room_latent_dim': 64, 'position_condition': False, 'position_emb_dim': 0, 'time_num': 1000, 'diffusion_semantic_kwargs': {'att_1': 0.99999, 'att_T': 9e-06, 'ctt_1': 9e-06, 'ctt_T': 0.99999, 'model_output_type': 'x0', 'mask_weight': 1, 'auxiliary_loss_weight': 0.0005, 'adaptive_auxiliary_loss': True}, 'diffusion_geometric_kwargs': {'schedule_type': 'linear', 'beta_start': 0.0001, 'beta_end': 0.02, 'loss_type': 'mse', 'model_mean_type': 'eps', 'model_var_type': 'fixedsmall', 'train_stats_file': '/mnt/sv-share/MiData/bedroom/dataset_stats.txt'}, 'net_type': 'transformer', 'net_kwargs': {'seperate_all': True, 'n_layer': 8, 'n_embd': 512, 'n_head': 4, 'dim_feedforward': 2048, 'dropout': 0.1, 'activate': 'GELU', 'timestep_type': 'adalayernorm_abs', 'mlp_type': 'fc'}, 'class_dim': 22}, 'feature_extractor': {'name': 'pointnet_simple', 'feat_units': [4, 64, 64, 512, 64]}, 'training': {'splits': ['overfit'], 'epochs': 50000, 'batch_size': 64, 'save_frequency': 100, 'max_grad_norm': 10, 'optimizer': 'Adam', 'weight_decay': 0.0, 'schedule': 'step', 'lr': 0.0002, 'lr_step': 10000, 'lr_decay': 0.5}, 'validation': {'splits': ['overfit'], 'frequency': 100, 'batch_size': 64}, 'logger': {'type': 'wandb', 'project': 'MiDiffusion'}}




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
        version = cfg.get("checkpoint_version", None)
        if version is not None and isinstance(version, int):
            checkpoint_path = download_version_checkpoint(
                run_path=run_path, version=version, download_dir=download_dir
            )
        else:
            checkpoint_path = download_latest_or_best_checkpoint(
                run_path=run_path, download_dir=download_dir, use_best=cfg.get("use_best", False)
            )
    else:
        # Use local path.
        checkpoint_path = Path(load_id)

    raw_train_dataset = get_raw_dataset(
        update_data_file_paths(config["data"], config),
        # config["data"], 
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=config["network"].get("room_mask_condition", True)
    ) 

    # Get Scaled dataset encoding (without data augmentation)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        update_data_file_paths(config["data"], config),
        # config["data"],
        split=config["validation"].get("splits", ["test"]),
        max_length=config["network"]["sample_num_points"],
        include_room_mask=config["network"].get("room_mask_condition", True)
    )

    # Create a CustomSceneDataset
    custom_dataset = CustomDataset(
        cfg=cfg.dataset,
        split=config["validation"].get("splits", ["test"]),
        ckpt_path=str(checkpoint_path)
    )

    # Save a ground truth sample from the dataset for comparison
    gt_sample_idx = 0  # Get the first sample
    gt_sample = custom_dataset[gt_sample_idx]
    
    gt_sample_np = gt_sample["scenes"].detach().cpu().numpy()
    gt_text_path = output_dir / "ground_truth_scene.txt"
    with open(gt_text_path, "w") as f:
        f.write(f"Ground truth scene shape: {gt_sample_np.shape}\n\n")
        f.write(f"Sample index: {gt_sample_idx}\n\n")
        f.write(np.array2string(gt_sample_np, threshold=np.inf, precision=6))
    
    logging.info(f"Saved ground truth scene to {gt_text_path}")
    
    # Create a dataloader for the custom dataset
    dataloader = torch.utils.data.DataLoader(
        custom_dataset,
        batch_size=num_scenes,
        num_workers=4,
        shuffle=False,
        persistent_workers=False,
        pin_memory=cfg.experiment.test.pin_memory,
    )
    
    print(f"[DEBUG] Created custom dataset with size: {len(custom_dataset)}")
    
    # Build experiment with custom dataset
    experiment = build_experiment(cfg, ckpt_path=checkpoint_path)
    
    try:
        print("[DEBUG] Starting to sample scenes...")
        # Sample scenes from the model
        sampled_scene_batches = experiment.exec_task("predict", dataloader=dataloader)
        #TODO: get indices of sampled scenes
        sampled_indices = list(range(len(sampled_scene_batches[0])))
        sampled_scenes = torch.cat(sampled_scene_batches, dim=0)
        
        print(f"[DEBUG] Sampled scenes shape: {sampled_scenes.shape}")
        
        sampled_scenes_np = sampled_scenes.detach().cpu().numpy() # b, 12, 30
        bbox_params_list = []
        n_classes = 22 #TODO: make it configurable, it should include empty token
        path_to_results = output_dir / "sampled_scenes_results.pkl" 
        for i in range(sampled_scenes_np.shape[0]):
            class_labels, translations, sizes, angles = [], [], [], []
            for j in range(sampled_scenes_np.shape[1]):
                class_label_idx = np.argmax(sampled_scenes_np[i,j,:n_classes])
                if class_label_idx != n_classes-1: #ignore if empty token
                    
                    class_labels.append(sampled_scenes_np[i,j,:n_classes])
                    translations.append(sampled_scenes_np[i,j,n_classes:n_classes+3])
                    sizes.append(sampled_scenes_np[i,j,n_classes+3:n_classes+6])
                    angles.append(sampled_scenes_np[i,j,n_classes+6:n_classes+8])
            bbox_params_list.append({
                "class_labels": np.array(class_labels)[None,:], 
                "translations": np.array(translations)[None,:],
                "sizes": np.array(sizes)[None,:],
                "angles": np.array(angles)[None,:]
            })
        # print("bbox param list", bbox_params_list)

        layout_list = []
        for bbox_params_dict in bbox_params_list:
            boxes = encoded_dataset.post_process(bbox_params_dict)
            bbox_params = {k: v[0] for k, v in boxes.items()}
            layout_list.append(bbox_params)

        # print("final output: ", layout_list)
        #layout_list [{"class_labels":[], "translations":[1,2,3], "sizes": [1,2,3,], "angles": [1]}, ...]
        threed_front_results = ThreedFrontResults(
        raw_train_dataset, raw_dataset, config, sampled_indices, layout_list
    )
    
        pickle.dump(threed_front_results, open(path_to_results, "wb"))
        print("Saved result to:", path_to_results)
        
        #TODO: fixme
        # kl_divergence = threed_front_results.kl_divergence()
        # print("object category kl divergence:", kl_divergence)
        
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
            if 'sampled_scenes' in locals():
                sampled_scenes_np = sampled_scenes.detach().cpu().numpy()
                text_path = output_dir / "partial_sampled_scenes.txt"
                with open(text_path, "w") as f:
                    f.write(f"Partial sampled scenes shape: {sampled_scenes_np.shape}\n\n")
                    f.write(np.array2string(sampled_scenes_np, threshold=np.inf, precision=6))
                logging.info(f"Saved partial sampled scenes to {text_path}")
                wandb.save(str(text_path))
        except:
            pass
        
        raise e

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()