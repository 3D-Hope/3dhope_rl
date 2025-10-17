#!/bin/bash
#SBATCH --job-name=rl_training
#SBATCH --partition=batch
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Exit on error but with better error reporting
set -e
trap 'echo "❌ Error on line $LINENO. Exit code: $?" >&2' ERR

# Create logs directory if it doesn't exist
mkdir -p logs

# Print debug information
echo "════════════════════════════════════════════════════════════════════════════════"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "System information:"
free -h
df -h /scratch/pramish_paudel
echo ""

echo "System information:"
free -h
df -h /scratch/pramish_paudel
echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 1: Copy model checkpoint
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 1: Checking model checkpoint..."
if [ ! -f "/scratch/pramish_paudel/model.ckpt" ]; then
    echo "Copying model checkpoint..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/model.ckpt /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy model checkpoint"
        exit 1
    }
    echo "✅ Model checkpoint copied"
else
    echo "✅ Model checkpoint already exists in scratch"
fi
echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 2: Copy and extract dataset
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 2: Checking bedroom dataset..."
if [ ! -d "/scratch/pramish_paudel/bedroom" ]; then
    echo "Copying bedroom dataset..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/bedroom.zip /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy bedroom dataset"
        exit 1
    }

    echo "Extracting dataset..."
    cd /scratch/pramish_paudel/
    unzip -q bedroom.zip || {
        echo "❌ Failed to extract bedroom dataset"
        exit 1
    }
    rm bedroom.zip
    echo "✅ Bedroom dataset extracted"
else
    echo "✅ Bedroom dataset already exists in scratch"
fi
echo ""

echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 3: Setup Miniconda/Miniforge
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 3: Setting up Conda environment..."
CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"

if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniforge..."
    mkdir -p /scratch/pramish_paudel/tools/
    cd /scratch/pramish_paudel/tools/

    # Download Miniforge
    wget -q --show-progress "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniforge.sh || {
        echo "❌ Failed to download Miniforge"
        exit 1
    }

    # Install silently
    bash miniforge.sh -b -p $CONDA_DIR || {
        echo "❌ Failed to install Miniforge"
        exit 1
    }
    rm miniforge.sh
    echo "✅ Miniforge installed at $CONDA_DIR"
else
    echo "✅ Miniforge already exists at $CONDA_DIR"
fi

# Source conda
echo "Sourcing conda..."
source "$CONDA_DIR/etc/profile.d/conda.sh" || {
    echo "❌ Failed to source conda"
    exit 1
}
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
echo ""

echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 4: Create and activate Python environment
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 4: Setting up Python environment..."
if ! conda env list | grep -q "3dhope_rl"; then
    echo "Creating conda environment: 3dhope_rl"
    conda create -n 3dhope_rl python=3.10 -y || {
        echo "❌ Failed to create conda environment"
        exit 1
    }

    echo "Activating conda environment: 3dhope_rl"
    conda activate 3dhope_rl || {
        echo "❌ Failed to activate conda environment"
        exit 1
    }

    echo "Installing pip in conda environment..."
    conda install pip -y || {
        echo "❌ Failed to install pip"
        exit 1
    }

    echo "✅ Environment setup complete"
else
    echo "✅ Environment '3dhope_rl' already exists"
    conda activate 3dhope_rl || {
        echo "❌ Failed to activate existing conda environment"
        exit 1
    }
fi

# Verify setup
echo ""
echo "Environment verification:"
echo "  Active conda environment: $CONDA_DEFAULT_ENV"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version)"
echo "  Pip path: $(which pip)"
echo ""
echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 5: Check GPU
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 5: Checking GPU availability..."
nvidia-smi || {
    echo "⚠️  GPU check failed, but continuing..."
}
echo ""

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 6: Setup project directory and dependencies
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 6: Installing project dependencies..."
cd ~/codes/3dhope_rl/ || {
    echo "❌ Failed to change to project directory"
    exit 1
}

echo "Current directory: $(pwd)"
echo ""

# Check for Poetry in multiple locations
POETRY_CMD=""
POETRY_HOME="/scratch/pramish_paudel/tools/poetry"
POETRY_BIN="$POETRY_HOME/bin/poetry"

# First, check if Poetry is available in current conda environment
echo "Checking for Poetry installation..."
if command -v poetry &> /dev/null; then
    POETRY_CMD="poetry"
    echo "✅ Poetry found in current environment: $(which poetry)"
    echo "   Version: $(poetry --version)"
# Second, check if Poetry exists in scratch
elif [ -f "$POETRY_BIN" ]; then
    POETRY_CMD="$POETRY_BIN"
    export PATH="$POETRY_HOME/bin:$PATH"
    echo "✅ Poetry found in scratch: $POETRY_BIN"
    echo "   Version: $($POETRY_BIN --version)"
# Third, try to install Poetry via conda (fastest and most reliable in conda env)
else
    echo "Poetry not found, trying to install via conda..."
    conda install -y -c conda-forge poetry 2>&1 | grep -v "Collecting package metadata" || {
        echo "⚠️  Conda install failed, installing to scratch..."
        mkdir -p "$POETRY_HOME"
        
        # Download and install Poetry to scratch
        echo "Downloading Poetry installer..."
        curl -sSL https://install.python-poetry.org | POETRY_HOME="$POETRY_HOME" python3 - || {
            echo "❌ Failed to install Poetry to scratch"
            echo "Falling back to pip installation..."
            pip install poetry || {
                echo "❌ Failed to install Poetry via pip"
                exit 1
            }
            POETRY_CMD="poetry"
        }
        
        if [ -f "$POETRY_BIN" ]; then
            POETRY_CMD="$POETRY_BIN"
            export PATH="$POETRY_HOME/bin:$PATH"
            echo "✅ Poetry installed to $POETRY_HOME"
        fi
    }
    
    # Check again after installation
    if command -v poetry &> /dev/null; then
        POETRY_CMD="poetry"
        echo "✅ Poetry installed successfully"
        echo "   Location: $(which poetry)"
        echo "   Version: $(poetry --version)"
    elif [ -f "$POETRY_BIN" ]; then
        POETRY_CMD="$POETRY_BIN"
        echo "✅ Poetry installed to scratch"
        echo "   Version: $($POETRY_BIN --version)"
    else
        echo "❌ Failed to install Poetry"
        exit 1
    fi
fi

echo ""

# Configure Poetry to create virtualenv in project
echo "Configuring Poetry..."
$POETRY_CMD config virtualenvs.in-project true || {
    echo "⚠️  Failed to configure Poetry, but continuing..."
}

# Install project dependencies
echo "Installing dependencies with Poetry..."
$POETRY_CMD install --no-interaction 2>&1 | tee /tmp/poetry_install.log || {
    echo "❌ Poetry install failed, falling back to pip..."
    echo "Last 20 lines of Poetry install log:"
    tail -20 /tmp/poetry_install.log
    
    pip install -e . || {
        echo "❌ Pip install also failed"
        exit 1
    }
    pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"
}

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating Poetry virtualenv..."
    source .venv/bin/activate
    echo "✅ Using Poetry virtualenv"
else
    echo "⚠️  No .venv found, using conda environment"
fi

# Login to wandb (use --relogin to avoid interactive prompt)
echo "Logging in to wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY" || echo "⚠️  wandb login failed, but continuing..."
else
    echo "⚠️  WANDB_API_KEY not set, skipping wandb login"
    echo "   Note: If you have wandb configured, it should work automatically"
fi

# Install ThreedFront package
echo "Installing ThreedFront package..."
pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"

# Set environment variables
export PYTHONUNBUFFERED=1
export DISPLAY=:0

echo "✅ All dependencies installed and configured"
# 🚀 Run training
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "STAGE 7: Starting RL training..."
echo "Training started at: $(date)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

export PYTHONUNBUFFERED=1

PYTHONPATH=. python -u main.py +name=baseline_with_dynamic_constraint_reward_tv_reg_50 \
    load=/scratch/pramish_paudel/model.ckpt \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/scratch/pramish_paudel/ \
    dataset.data.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_flux_transformer \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2000000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=50.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=256 \
    experiment.training.checkpointing.every_n_train_steps=500 \
    algorithm.num_additional_tokens_for_sampling=0 \
    algorithm.ddpo.n_timesteps_to_sample=100 \
    experiment.find_unused_parameters=True \
    algorithm.custom.loss=true \
    algorithm.validation.num_samples_to_render=0 \
    algorithm.validation.num_samples_to_visualize=0 \
    algorithm.validation.num_directives_to_generate=0 \
    algorithm.test.num_samples_to_render=0 \
    algorithm.test.num_samples_to_visualize=0 \
    algorithm.test.num_directives_to_generate=0 \
    algorithm.validation.num_samples_to_compute_physical_feasibility_metrics_for=0 \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    algorithm.ddpo.dynamic_constraint_rewards.reward_code_dir=/home/pramish_paudel/codes/3dhope_rl/dynamic_constraint_rewards/dynamic_reward_functions \
    algorithm.ddpo.dynamic_constraint_rewards.stats_path=/home/pramish_paudel/codes/3dhope_rl/dynamic_constraint_rewards/stats.json \
    algorithm.ddpo.dynamic_constraint_rewards.room_type=bedroom \
    experiment.training.precision=bf16-mixed \
    experiment.validation.precision=bf16-mixed \
    experiment.test.precision=bf16-mixed \
    experiment.matmul_precision=medium

    

# Check exit status
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully at: $(date)"
    exit 0
else
    echo "❌ Training failed at: $(date)"
    exit 1
fi
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "Job completed at: $(date)"
