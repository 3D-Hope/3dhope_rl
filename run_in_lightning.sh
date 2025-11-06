# run in ajad pc
# Define your SSH destination
# ssh_machine="s_01k9btwd5h51p4e1qsrv1anxcg@ssh.lightning.ai"
# remote_dir="/teamspace/studios/this_studio"

# # # Upload all three files via scp
# scp -v \
#   /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_accessibility_cache.zip \
#   /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom_sdf_cache.zip \
#   "$ssh_machine:$remote_dir/"

# scp -v \
#   /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/bedroom.zip \
#   "$ssh_machine:$remote_dir/"

# run in lightning pc
# unzip all
# git clone https://github.com/3D-Hope/3dhope_rl/
# git clone https://github.com/3D-Hope/ThreedFront/


# # Install project dependencies
# echo "Installing dependencies with Poetry..."
# $POETRY_CMD install --no-interaction 2>&1 | tee /tmp/poetry_install.log || {
#     echo "❌ Poetry install failed, falling back to pip..."
#     echo "Last 20 lines of Poetry install log:"
#     tail -20 /tmp/poetry_install.log
    
#     pip install -e . || {
#         echo "❌ Pip install also failed"
#         exit 1
#     }
#     pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"
# }


# Install project dependencies
# echo "Installing dependencies with Poetry..."
# $POETRY_CMD install --no-interaction 2>&1 | tee /tmp/poetry_install.log || {
#     echo "❌ Poetry install failed, falling back to pip..."
#     echo "Last 20 lines of Poetry install log:"
#     tail -20 /tmp/poetry_install.log
    
#     pip install -e . || {
#         echo "❌ Pip install also failed"
#         exit 1
#     }
#     pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"
# }

# # Activate virtual environment
# if [ -d ".venv" ]; then
#     echo "Activating Poetry virtualenv..."
#     source .venv/bin/activate
#     echo "✅ Using Poetry virtualenv"
# else
#     echo "⚠️  No .venv found, using conda environment"
# fi

# # Login to wandb (use --relogin to avoid interactive prompt)
# echo "Logging in to wandb..."
# if [ -n "$WANDB_API_KEY" ]; then
#     wandb login --relogin "$WANDB_API_KEY" || echo "⚠️  wandb login failed, but continuing..."
# else
#     echo "⚠️  WANDB_API_KEY not set, skipping wandb login"
#     echo "   Note: If you have wandb configured, it should work automatically"
# fi

# # Install ThreedFront package
# echo "Installing ThreedFront package..."
# pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"

# # Set environment variables
# export PYTHONUNBUFFERED=1
# export DISPLAY=:0

# echo "✅ All dependencies installed and configured"
# # 🚀 Run training
# echo ""
# echo "════════════════════════════════════════════════════════════════════════════════"
# echo "STAGE 7: Starting RL training..."
# echo "Training started at: $(date)"
# echo "════════════════════════════════════════════════════════════════════════════════"
# echo ""

# export PYTHONUNBUFFERED=1

# PYTHONPATH=. python -u main.py +name=universal_bedroom \
#     load=rrudae6n \
#     dataset=custom_scene \
#     dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
#     dataset.data.path_to_processed_data=/teamspace/studios/this_studio/3dhope_rl/ \
#     dataset.data.path_to_dataset_files=/teamspace/studios/this_studio/ThreedFront/dataset_files \
#     dataset.max_num_objects_per_scene=12 \
#     algorithm=scene_diffuser_midiffusion \
#     algorithm.classifier_free_guidance.use=False \
#     algorithm.ema.use=True \
#     algorithm.trainer=rl_score \
#     algorithm.noise_schedule.scheduler=ddim \
#     experiment.training.max_steps=1020000 \
#     experiment.validation.limit_batch=1 \
#     experiment.validation.val_every_n_step=50 \
#     algorithm.ddpo.ddpm_reg_weight=50.0 \
#     experiment.reset_lr_scheduler=True \
#     experiment.training.lr=1e-6 \
#     experiment.lr_scheduler.num_warmup_steps=250 \
#     algorithm.ddpo.batch_size=128 \
#     experiment.training.checkpointing.every_n_train_steps=500 \
#     algorithm.num_additional_tokens_for_sampling=0 \
#     algorithm.ddpo.n_timesteps_to_sample=100 \
#     experiment.find_unused_parameters=True \
#     algorithm.custom.loss=True \
#     algorithm.validation.num_samples_to_render=0 \
#     algorithm.validation.num_samples_to_visualize=0 \
#     algorithm.validation.num_directives_to_generate=0 \
#     algorithm.test.num_samples_to_render=0 \
#     algorithm.test.num_samples_to_visualize=0 \
#     algorithm.test.num_directives_to_generate=0 \
#     algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
#     algorithm.ddpo.use_universal_reward=True \
#     experiment.training.precision=bf16-mixed \
#     experiment.validation.precision=bf16-mixed \
#     experiment.test.precision=bf16-mixed \
#     experiment.matmul_precision=medium \
#     algorithm.classifier_free_guidance.use_floor=True \
#     algorithm.ddpo.dynamic_constraint_rewards.stats_path=/teamspace/studios/this_studio/3dhope_rl/dynamic_constraint_rewards/stats.json \
#     dataset.sdf_cache_dir=/teamspace/studios/this_studio/3dhope_rl/living_sdf_cache/ \
#     dataset.accessibility_cache_dir=/teamspace/studios/this_studio/3dhope_rl/living_accessibility_cache/ \
#     algorithm.custom.num_classes=22 \
#     algorithm.custom.objfeat_dim=0 \
#     algorithm.custom.obj_vec_len=30 \
#     algorithm.custom.obj_diff_vec_len=30 \
#     dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
#     dataset.data.dataset_directory=bedroom \
#     dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
#     algorithm.custom.old=True \
#     dataset.data.room_type=bedroom



#!/bin/bash
set -e

echo "🚀 Unzipping uploaded files..."
unzip -o bedroom_accessibility_cache.zip
unzip -o bedroom_sdf_cache.zip
unzip -o bedroom.zip

echo "✅ Unzipping done."

# Clone necessary repositories
echo "🚀 Cloning repositories..."
# git clone https://github.com/3D-Hope/3dhope_rl/
git clone https://github.com/3D-Hope/ThreedFront/

# Install project dependencies
echo "🚀 Installing dependencies with Poetry..."
POETRY_CMD=$(which poetry || echo "poetry")
$POETRY_CMD install --no-interaction 2>&1 | tee /tmp/poetry_install.log || {
    echo "❌ Poetry install failed, falling back to pip..."
    tail -20 /tmp/poetry_install.log

    pip install -e . || {
        echo "❌ Pip install also failed."
        exit 1
    }

    pip install -e ../ThreedFront || echo "⚠️ ThreedFront install failed"
}

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating Poetry virtualenv..."
    source .venv/bin/activate
    echo "✅ Using Poetry virtualenv"
else
    echo "⚠️ No .venv found, using conda or system env"
fi

# Login to wandb (optional)
echo "🚀 Logging into wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY" || echo "⚠️ wandb login failed, continuing..."
else
    echo "⚠️ WANDB_API_KEY not set, skipping wandb login."
fi

# Install ThreedFront package
echo "🚀 Installing ThreedFront package..."
pip install -e ../ThreedFront || echo "⚠️ ThreedFront install failed."

# Set environment variables
export PYTHONUNBUFFERED=1
export DISPLAY=:0

echo "✅ All dependencies installed and configured."

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STAGE 7: Starting RL training..."
echo "Training started at: $(date)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

PYTHONPATH=. python -u main.py +name=universal_bedroom \
    load=rrudae6n \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/teamspace/studios/this_studio/3dhope_rl/ \
    dataset.data.path_to_dataset_files=/teamspace/studios/this_studio/ThreedFront/dataset_files \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    experiment.training.max_steps=1020000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=128 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=True \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.ddpo.use_universal_reward=True \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium \
    algorithm.classifier_free_guidance.use_floor=True \
    algorithm.ddpo.dynamic_constraint_rewards.stats_path=/teamspace/studios/this_studio/3dhope_rl/dynamic_constraint_rewards/stats.json \
    dataset.sdf_cache_dir=/teamspace/studios/this_studio/3dhope_rl/bedroom_sdf_cache/ \
    dataset.accessibility_cache_dir=/teamspace/studios/this_studio/3dhope_rl/bedroom_accessibility_cache/ \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    algorithm.custom.old=True \
    dataset.data.room_type=bedroom
