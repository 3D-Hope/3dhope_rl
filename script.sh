#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --partition=debug
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Create logs directory if it doesn't exist
# mkdir -p logs

# Print debug information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Scratch contents: $(ls /scratch/pramish_paudel)"
free -h
df -h
echo "copying data "

# Once you get the interactive shell, run:
rsync -aHzv /home/pramish_paudel/3dhope_data/bedroom.zip /scratch/pramish_paudel/
rsync -aHzv /home/pramish_paudel/3dhope_data/model.ckpt /scratch/pramish_paudel/


# Then unzip
cd /scratch/pramish_paudel/
unzip bedroom.zip

# Remove the zip file
rm bedroom.zip
# 🔧 Setup Miniconda
CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"

if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniconda/Miniforge..."
    mkdir -p /scratch/pramish_paudel/tools/
    cd /scratch/pramish_paudel/tools/

    # Download Miniforge (better than Miniconda for conda-forge packages)
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m)>

    # Install silently
    bash miniforge.sh -b -p $CONDA_DIR
    rm miniforge.sh
    echo "✅ Miniforge installed at $CONDA_DIR"
else
    echo "✅ Miniforge already exists at $CONDA_DIR"
fi

# Source conda
echo "Sourcing conda from: $CONDA_DIR/etc/profile.d/conda.sh"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Initialize conda for this session
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# 🐍 Create and setup environment
if ! conda env list | grep -q "midiffusion"; then
    echo "Creating conda environment: midiffusion"
    conda create -n midiffusion python=3.10 -y

    echo "Activating conda environment: midiffusion"
    conda activate midiffusion
    
    echo "Installing conda packages..."
    conda install -c conda-forge numpy scipy pyyaml tqdm matplotlib scikit-learn seaborn pillow opencv wxpy>

    echo "Installing PyTorch with CUDA 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    echo "Installing remaining packages..."
    pip install clean-fid pyrr shapely simple-3dviz trimesh einops wandb

    echo "✅ Environment setup complete!"
else
    echo "✅ Environment 'midiffusion' already exists"
    conda activate midiffusion
fi
# pip install dotenv
# Verify setup
echo "Active conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Test PyTorch installation
echo "Testing PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}'); print(f'>device: {torch.cuda.get_device_name(0)}')"

# Check GPU
echo "GPU information:"
nvidia-smi

# 📂 Move to your training script directory
cd ~/codes/3dhope_rl/

# Verify we're in the right directory
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la

pip install -e ../ThreedFront


curl -sSL https://install.python-poetry.org | python3 -

poetry config virtualenvs.in-project true

poetry install

source .venv/bin/activate

wandb login

pip install -e ../ThreedFront


# # install midiffusion
# python setup.py build_ext --inplace
# pip install -e .
# Check if config file exists
# if [ ! -f "config/bedrooms_mixed.yaml" ]; then
#     echo "⚠️  WARNING: config/bedrooms_mixed.yaml not found!"
#     echo "Available config files:"
#     ls -la config/ 2>/dev/null || echo "No config directory found"
# fi

# # Check if training script exists
# if [ ! -f "scripts/train_diffusion.py" ]; then
#     echo "⚠️  WARNING: scripts/train_ndiffusion.py not found!"
#     echo "Available scripts:"
#     ls -la scripts/ 2>/dev/null || echo "No scripts directory found"
# fi

# 🚀 Run training
echo "Starting training at: $(date)"
export PYTHONUNBUFFERED=1
PYTHONPATH=. python -u main.py +name=first_rl \
load=/home/pramish_paudel/3dhope_data/model.ckpt \
dataset=custom_scene \
dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
dataset.path_to_processed_data=/scratch/pramish_paudel/bedroom/ \
dataset.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
dataset.max_num_objects_per_scene=12 \
algorithm=scene_diffuser_flux_transformer \
algorithm.classifier_free_guidance.use=False \
algorithm.ema.use=False \
algorithm.trainer=rl_score \
algorithm.ddpo.use_iou_reward=True \
algorithm.ddpo.use_object_number_reward=False \
algorithm.noise_schedule.scheduler=ddim \
algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
experiment.training.max_steps=2e6 \
experiment.validation.limit_batch=1 \
experiment.validation.val_every_n_step=50 \
algorithm.ddpo.ddpm_reg_weight=200.0 \
experiment.reset_lr_scheduler=True \
experiment.training.lr=1e-6 \
experiment.lr_scheduler.num_warmup_steps=250 \
algorithm.ddpo.batch_size=32 \
experiment.training.checkpointing.every_n_train_steps=500 \
algorithm.num_additional_tokens_for_sampling=2 \
algorithm.ddpo.n_timesteps_to_sample=100 \
experiment.find_unused_parameters=True \
algorithm.custom.loss=true
# Check exit status
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully at: $(date)"
else
    echo "❌ Training failed at: $(date)"
fi

echo "Job completed at: $(date)"


