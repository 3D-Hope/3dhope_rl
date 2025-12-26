source .venv/bin/activate #mbysryxi


# PYTHONPATH=. python -u scripts/analyze_scene_parameters.py +num_scenes=10 \
#     +intermediate_steps=[10,50,75,100,125] \
#     +num_quick_steps=15 \
#     +save_scenes=false \
#     load=gtjphzpb \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=true \
#     algorithm.trainer=rl_score \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     algorithm.classifier_free_guidance.use_floor=true \
#     algorithm.custom.old=False \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     wandb.mode=disabled \
#     algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
#     algorithm.ddpo.dynamic_constraint_rewards.use=True \
#     dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
#     dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \
    


# PYTHONPATH=. python -u scripts/test_intermediate_rewards.py +num_scenes=1000 \
# PYTHONPATH=. python -u scripts/custom_sample_and_render.py +num_scenes=100 \
#     +intermediate_steps=[10,50,75,100,125] \
#     +num_quick_steps=15 \
#     +save_scenes=false \
#     load=lpm71nm1 \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=true \
#     algorithm.trainer=ddpm \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     algorithm.classifier_free_guidance.use_floor=true \
#     algorithm.custom.old=False \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     wandb.mode=disabled \
#     algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
#     algorithm.ddpo.dynamic_constraint_rewards.use=True \
#     dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
#     dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \
    

# PYTHONPATH=. python -u scripts/visualize_forward_diffusion.py +num_scenes=1 \
#     load=gtjphzpb \
#     +num_timesteps=150 \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=true \
#     algorithm.trainer=rl_score \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     algorithm.classifier_free_guidance.use_floor=true \
#     algorithm.custom.old=False \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     wandb.mode=disabled \
#     algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
#     algorithm.ddpo.dynamic_constraint_rewards.use=True \
#     dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
#     dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \

    # checkpoint_version=15


# python ../ThreedFront/scripts/render_results.py --no_texture --retrieve_by_size /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-24/02-19-49/sampled_scenes_results.pkl


# checkpoint_version=7 \
# PYTHONPATH=. python -u scripts/custom_sample_and_render.py +num_scenes=1000 \
# load=xvthawzz \
# dataset=custom_scene \
# dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
# dataset.max_num_objects_per_scene=12 \
# algorithm=scene_diffuser_midiffusion \
# algorithm.trainer=ddpm \
# algorithm.noise_schedule.scheduler=ddim \
# algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
# experiment.find_unused_parameters=True \
# algorithm.classifier_free_guidance.use=False \
# algorithm.classifier_free_guidance.use_floor=True \
# algorithm.classifier_free_guidance.weight=0 \
# algorithm.custom.loss=true \
# algorithm.ema.use=True \
# dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm_no_prm \
# algorithm.custom.old=False \
# wandb.mode=disabled \
# experiment.test.batch_size=256 \
# checkpoint_version=20



# PYTHONPATH=. python -u main.py +name=test \
# PYTHONPATH=. python -u scripts/custom_sample_and_render.py +num_scenes=1000 \
#     load=gtjphzpb \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=True \
#     algorithm.trainer=ddpm \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     algorithm.classifier_free_guidance.use_floor=True \
#     algorithm.custom.old=False \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
#     wandb.mode=disabled


# PYTHONPATH=. python -u main.py +name=test \
#     load=rrudae6n \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=False \
#     algorithm.trainer=rl_score \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     experiment.training.max_steps=2e6 \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     algorithm.ddpo.ddpm_reg_weight=50.0 \
#     experiment.reset_lr_scheduler=True \
#     experiment.training.lr=1e-6 \
#     experiment.lr_scheduler.num_warmup_steps=250 \
#     algorithm.ddpo.batch_size=4 \
#     experiment.training.batch_size=4 \
#     experiment.validation.batch_size=4 \
#     experiment.test.batch_size=4 \
#     experiment.training.checkpointing.every_n_train_steps=500 \
#     algorithm.num_additional_tokens_for_sampling=0 \
#     algorithm.ddpo.n_timesteps_to_sample=10 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     experiment.training.precision=bf16-mixed \
#     algorithm.ddpo.use_universal_reward=True \
#     algorithm.classifier_free_guidance.use_floor=True \
#     algorithm.classifier_free_guidance.weight=1.0 \
#     algorithm.custom.old=False \
#     algorithm.ddpo.use_inpaint=False \
#     algorithm.ddpo.dynamic_constraint_rewards.stats_path=dynamic_constraint_rewards/stats.json \
#     dataset.sdf_cache_dir=./bedroom_sdf_cache/ \
#     dataset.accessibility_cache_dir=./bedroom_accessibility_cache/ \
#     wandb.mode=disabled




# python scripts/custom_sample_and_render.py +num_scenes=1000 load=fhfnf4xi     dataset=custom_scene     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json     dataset.max_num_objects_per_scene=12     algorithm=scene_diffuser_midiffusion     algorithm.classifier_free_guidance.use=False     algorithm.ema.use=True     algorithm.trainer=ddpm     algorithm.noise_schedule.scheduler=ddim     algorithm.noise_schedule.ddim.num_inference_timesteps=150     experiment.training.max_steps=1020000     experiment.validation.limit_batch=1     experiment.validation.val_every_n_step=50     algorithm.ddpo.ddpm_reg_weight=40     experiment.reset_lr_scheduler=True     experiment.training.lr=1e-6     experiment.lr_scheduler.num_warmup_steps=250     algorithm.ddpo.batch_size=128     experiment.training.checkpointing.every_n_train_steps=500     algorithm.num_additional_tokens_for_sampling=0     algorithm.ddpo.n_timesteps_to_sample=100     experiment.find_unused_parameters=True     algorithm.custom.loss=True     algorithm.validation.num_samples_to_render=0     algorithm.validation.num_samples_to_visualize=0     algorithm.validation.num_directives_to_generate=0     algorithm.test.num_samples_to_render=0     algorithm.test.num_samples_to_visualize=0     algorithm.test.num_directives_to_generate=0     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0     algorithm.ddpo.use_universal_reward=True     algorithm.classifier_free_guidance.use_floor=True     algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/dynamic_constraint_rewards     algorithm.ddpo.dynamic_constraint_rewards.user_query=universal_bedroom     dataset.sdf_cache_dir=./bedroom_sdf_cache/     dataset.accessibility_cache_dir=./bedroom_accessibility_cache/     algorithm.custom.num_classes=22     algorithm.custom.objfeat_dim=0     algorithm.custom.obj_vec_len=30     algorithm.custom.obj_diff_vec_len=30     


# PYTHONPATH=. python main.py +name=flux_floor \
# PYTHONPATH=. python scripts/custom_sample_and_render.py \
# +num_scenes=1000 \
# load=ca0l19rv \
# dataset=custom_scene \
# dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
# dataset._name=custom_scene \
# dataset.max_num_objects_per_scene=12 \
# algorithm=scene_diffuser_flux_transformer \
# algorithm.trainer=ddpm \
# experiment.find_unused_parameters=True \
# algorithm.classifier_free_guidance.use=False \
# algorithm.classifier_free_guidance.use_floor=True \
# algorithm.classifier_free_guidance.weight=0 \
# algorithm.custom.loss=true \
# algorithm.ema.use=True \
# debug=True \
# wandb.mode=disabled \
# algorithm.noise_schedule.scheduler=ddim \
# algorithm.noise_schedule.ddim.num_inference_timesteps=150
# # # 
# checkpoint_version=5 \

# checkpoint_version=21 \
PYTHONPATH=. python dynamic_constraint_rewards/compute_success_rates.py +num_scenes=1000 \
load=nots1b42 \
dataset=custom_scene \
algorithm=scene_diffuser_midiffusion \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.max_num_objects_per_scene=12 \
experiment.test.batch_size=256 \
algorithm.trainer=ddpm \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.find_unused_parameters=True \
algorithm.classifier_free_guidance.use=False \
algorithm.classifier_free_guidance.use_floor=True \
algorithm.classifier_free_guidance.weight=0 \
algorithm.custom.loss=true \
algorithm.ema.use=true \
dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm_no_prm \
algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
algorithm.ddpo.dynamic_constraint_rewards.use=True \
algorithm.custom.old=False \
wandb.mode=disabled \
# experiment.seed=21 \

# python ../ThreedFront/scripts/render_results.py --no_texture --retrieve_by_size /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-12-26/19-49-00/sampled_scenes_results.pkl


# PYTHONPATH=. python scripts/generate_and_save_trajectory.py +num_scenes=1000 \
#     load=gtjphzpb \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=True \
#     algorithm.trainer=ddpm \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=True \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     algorithm.classifier_free_guidance.use_floor=true \
#     algorithm.custom.old=False \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=10 \
#     +compute_losses=true \
#     wandb.mode=disabled




# PYTHONPATH=. python scripts/generate_and_save_trajectory.py +scene_idx=1 \
# load=pfksynuz \
# dataset=custom_scene \
# dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
# dataset.max_num_objects_per_scene=12 \
# experiment.test.batch_size=32 \
# algorithm=scene_diffuser_midiffusion \
# algorithm.trainer=ddpm \
# algorithm.noise_schedule.scheduler=ddim \
# algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
# experiment.find_unused_parameters=True \
# algorithm.classifier_free_guidance.use=False \
# algorithm.classifier_free_guidance.use_floor=False \
# algorithm.classifier_free_guidance.weight=0 \
# algorithm.custom.loss=true \
# algorithm.ema.use=True \
# dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm_no_prm \
# experiment.seed=21

# PYTHONPATH=. python scripts/generate_and_save_trajectory.py +num_scenes=16 \
# load=gtjphzpb \
# dataset=custom_scene \
# dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
# dataset.max_num_objects_per_scene=12 \
# experiment.test.batch_size=32 \
# algorithm=scene_diffuser_midiffusion \
# algorithm.trainer=ddpm \
# algorithm.noise_schedule.scheduler=ddim \
# algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
# experiment.find_unused_parameters=True \
# algorithm.classifier_free_guidance.use=False \
# algorithm.classifier_free_guidance.use_floor=True \
# algorithm.classifier_free_guidance.weight=0 \
# algorithm.custom.loss=true \
# algorithm.ema.use=True \
# dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm_no_prm \
# algorithm.custom.old=False
# experiment.seed=21

# python scripts/custom_sample_and_render.py \
#     load=xvthawzz \
#     checkpoint_version=5 \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     +num_scenes=1000 \
#     algorithm=scene_diffuser_midiffusion \
#     algorithm.trainer=ddpm \
#     experiment.find_unused_parameters=True \
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.classifier_free_guidance.use_floor=True \
#     algorithm.classifier_free_guidance.weight=1 \
#     algorithm.custom.loss=true \
#     algorithm.ema.use=True \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     algorithm.custom.objfeat_dim=0 \
#     algorithm.custom.old=false \
#     algorithm.custom.loss=True







# source .venv/bin/activate
# PYTHONPATH=. python -u main.py +name=test_inpaint_rl \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion\
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=False \
#     algorithm.trainer=rl_score \
#     algorithm.noise_schedule.scheduler=ddim \
#     algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
#     experiment.training.max_steps=2e6 \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     algorithm.ddpo.ddpm_reg_weight=50.0 \
#     experiment.reset_lr_scheduler=True \
#     experiment.training.lr=1e-6 \
#     experiment.lr_scheduler.num_warmup_steps=250 \
#     algorithm.ddpo.batch_size=4 \
#     experiment.training.checkpointing.every_n_train_steps=500 \
#     algorithm.num_additional_tokens_for_sampling=0 \
#     algorithm.ddpo.n_timesteps_to_sample=100 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     experiment.training.precision=bf16-mixed \
#     algorithm.ddpo.use_universal_reward=True \
#     algorithm.ddpo.universal_reward.use_physcene_reward=True \
#     algorithm.classifier_free_guidance.use_floor=True \
#     load=rrudae6n \
#     algorithm.classifier_free_guidance.weight=1.0 \
#     algorithm.custom.old=True \
#     algorithm.predict.inpaint_masks='{ceiling_lamp: 4}' \
#     algorithm.ddpo.use_inpaint=True \

# PYTHONPATH=. python -u main.py +name=continuous_mi_bedroom_floor_obj32 \
#     dataset=custom_scene \
#     dataset.data.dataset_directory=bedroom \
#     dataset.data.path_to_processed_data=/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/3d_front_data \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset._name=custom_scene \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion \
#     algorithm.trainer=ddpm \
#     algorithm.custom.objfeat_dim=32 \
#     experiment.find_unused_parameters=True \
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.classifier_free_guidance.use_floor=True \
#     algorithm.classifier_free_guidance.weight=0 \
#     algorithm.custom.obj_vec_len=62 \
#     algorithm.custom.obj_diff_vec_len=62 \
#     algorithm.custom.loss=true \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_objfeats_lat32_wocm
