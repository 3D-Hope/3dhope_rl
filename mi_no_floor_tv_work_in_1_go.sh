#!/bin/bash
#SBATCH --job-name=mi_no_floor_tv_work_in_1_go
#SBATCH --partition=batch
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=20G
#SBATCH --time=15:00:00
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
export WANDB_ENTITY="078bct021-ashok-d"

# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 1: Copy model checkpoint
# ═══════════════════════════════════════════════════════════════════════════════════
# echo "STAGE 1: Checking model checkpoint..."
# if [ ! -f "/scratch/pramish_paudel/model.ckpt" ]; then
#     echo "Copying model checkpoint..."
#     rsync -aHzv --progress /home/pramish_paudel/3dhope_data/model.ckpt /scratch/pramish_paudel/ || {
#         echo "❌ Failed to copy model checkpoint"
#         exit 1
#     }
#     echo "✅ Model checkpoint copied"
# else
#     echo "✅ Model checkpoint already exists in scratch"
# fi
# echo ""

# Move SDF cache
rm -rf /scratch/pramish_paudel/bedroom_sdf_cache
if [ ! -d "/scratch/pramish_paudel/bedroom_sdf_cache" ]; then
    echo "Copying SDF cache..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/bedroom_sdf_cache.zip /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy SDF cache"
        exit 1
    }

    echo "Extracting SDF cache..."
    unzip -o /scratch/pramish_paudel/bedroom_sdf_cache.zip -d /scratch/pramish_paudel/ || {
        echo "❌ Failed to extract SDF cache"
        exit 1
    }

    rm /scratch/pramish_paudel/bedroom_sdf_cache.zip
    echo "✅ SDF cache copied"
else
    echo "✅ SDF cache already exists in scratch"
fi

ls /scratch/pramish_paudel/bedroom_sdf_cache


rm -rf /scratch/pramish_paudel/bedroom_accessibility_cache
echo "Checking accessibility cache..."
if [ ! -d "/scratch/pramish_paudel/bedroom_accessibility_cache" ]; then
    echo "Copying accessibility cache..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/bedroom_accessibility_cache.zip /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy accessibility cache"
        exit 1
    }
    unzip -o /scratch/pramish_paudel/bedroom_accessibility_cache.zip -d /scratch/pramish_paudel/ || {
        echo "❌ Failed to extract accessibility cache"
        exit 1
    }
    rm /scratch/pramish_paudel/bedroom_accessibility_cache.zip
    echo "✅ accessibility cache copied"
else
    echo "✅ accessibility cache already exists in scratch"
fi
echo ""
ls /scratch/pramish_paudel/bedroom_accessibility_cache

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
    unzip -oq bedroom.zip || {
        echo "❌ Failed to extract bedroom dataset"
        exit 1
    }
    rm bedroom.zip
    echo "✅ bedroom dataset extracted"
else
    echo "✅ bedroom dataset already exists in scratch"
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
 =""
POETRY_HOME="/scratch/pramish_paudel/tools/poetry"
POETRY_BIN="$POETRY_HOME/bin/poetry"

# First, check if Poetry is available in current conda environment
echo "Checking for Poetry installation..."
# ═══════════════════════════════════════════════════════════════════════════════════
# STAGE 5: Poetry Installation and Configuration
# ═══════════════════════════════════════════════════════════════════════════════════
echo "STAGE 5: Poetry Setup"
echo "───────────────────────────────────────────────────────────────────────────────"

cd ~/codes/3dhope_rl/ || {
    echo "❌ Failed to change to project directory"
    exit 1
}

echo "Current directory: $(pwd)"
echo ""

# Define Poetry paths
POETRY_HOME="/scratch/pramish_paudel/tools/poetry"
POETRY_BIN="$POETRY_HOME/bin/poetry"

# Remove ~/.local/bin from PATH to avoid global Poetry
echo "🔧 Cleaning PATH to avoid conflicts..."
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "\.local/bin" | tr '\n' ':' | sed 's/:$//')

# Check if scratch Poetry exists
if [ -f "$POETRY_BIN" ]; then
    echo "✅ Found Poetry at: $POETRY_BIN"
else
    echo "📦 Installing Poetry to scratch..."
    mkdir -p "$POETRY_HOME"
    
    # Install Poetry to scratch location
    curl -sSL https://install.python-poetry.org | POETRY_HOME="$POETRY_HOME" python3 - || {
        echo "❌ Failed to install Poetry to scratch"
        exit 1
    }
    
    if [ ! -f "$POETRY_BIN" ]; then
        echo "❌ Poetry installation failed - binary not found"
        exit 1
    fi
    echo "✅ Poetry installed to $POETRY_HOME"
fi

# Add scratch Poetry to PATH (put it FIRST to override everything)
export PATH="$POETRY_HOME/bin:$PATH"

# Verify we're using the correct Poetry
POETRY_PATH=$(which poetry)
echo ""
echo "Poetry Information:"
echo "  Expected: $POETRY_BIN"
echo "  Actual:   $POETRY_PATH"
echo "  Version:  $(poetry --version)"

if [ "$POETRY_PATH" != "$POETRY_BIN" ]; then
    echo "❌ ERROR: Wrong Poetry is being used!"
    echo "   This might cause package installation issues"
    exit 1
fi
echo "  ✅ Confirmed: Using scratch Poetry"
echo ""

# Configure Poetry to use conda's Python (not create its own venv)
# Configure Poetry to use conda's Python (not create its own venv)
echo "🔧 Configuring Poetry to use conda environment..."
poetry config virtualenvs.create false
poetry config virtualenvs.in-project false

echo "📋 Poetry configuration:"
poetry config --list | grep virtualenvs

echo ""
echo "✅ STAGE 5 Complete: Poetry configured to use scratch installation"
echo ""
# Install project dependencies
POETRY_CMD=""
POETRY_HOME="/scratch/pramish_paudel/tools/poetry"
POETRY_BIN="$POETRY_HOME/bin/poetry"

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

# Get Python path from Poetry's virtualenv
echo "Getting Python executable from Poetry environment..."
PYTHON_CMD=$($POETRY_CMD env info -p 2>/dev/null)/bin/python

if [ -f "$PYTHON_CMD" ]; then
    echo "✅ Found Poetry Python at: $PYTHON_CMD"
    echo "   Python version: $($PYTHON_CMD --version)"
else
    echo "⚠️  Could not find Poetry Python, checking for .venv..."
    if [ -d ".venv" ]; then
        PYTHON_CMD=".venv/bin/python"
        echo "✅ Using .venv Python: $PYTHON_CMD"
    else
        PYTHON_CMD="python"
        echo "⚠️  Using system Python: $PYTHON_CMD"
    fi
fi

# Verify hydra is available
echo "Verifying required packages..."
$PYTHON_CMD -c "import hydra; print('✅ Hydra found')" || {
    echo "❌ Hydra not found in Poetry environment"
    exit 1
}

# Login to wandb (use --relogin to avoid interactive prompt)
echo "Logging in to wandb..."
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY" || echo "⚠️  wandb login failed, but continuing..."
else
    echo "⚠️  WANDB_API_KEY not set, skipping wandb login"
    echo "   Note: If you have wandb configured, it should work automatically"
fi

# Install ThreedFront package using Poetry's Python
echo "Installing ThreedFront package..."
$PYTHON_CMD -m pip install -e ../ThreedFront || echo "⚠️  ThreedFront install failed"

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

PYTHONPATH=. $PYTHON_CMD -u main.py +name=mi_no_floor_tv_work_in_1_go \
    load=pfksynuz \
    dataset=custom_scene \
    dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
    dataset.data.path_to_processed_data=/scratch/pramish_paudel/ \
    dataset.data.path_to_dataset_files=/home/pramish_paudel/codes/ThreedFront/dataset_files \
    dataset.max_num_objects_per_scene=12 \
    algorithm=scene_diffuser_midiffusion \
    algorithm.classifier_free_guidance.use=False \
    algorithm.ema.use=True \
    algorithm.trainer=rl_score \
    algorithm.noise_schedule.scheduler=ddim \
    algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
    experiment.training.max_steps=2000000 \
    experiment.validation.limit_batch=1 \
    experiment.validation.val_every_n_step=50 \
    algorithm.ddpo.ddpm_reg_weight=100.0 \
    experiment.reset_lr_scheduler=True \
    experiment.training.lr=1e-6 \
    experiment.lr_scheduler.num_warmup_steps=250 \
    algorithm.ddpo.batch_size=256 \
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
    algorithm.classifier_free_guidance.use_floor=false \
    algorithm.ddpo.dynamic_constraint_rewards.use=True \
    dataset.sdf_cache_dir=/scratch/pramish_paudel/bedroom_sdf_cache/ \
    dataset.accessibility_cache_dir=/scratch/pramish_paudel/bedroom_accessibility_cache/ \
    algorithm.custom.num_classes=22 \
    algorithm.custom.objfeat_dim=0 \
    algorithm.custom.obj_vec_len=30 \
    algorithm.custom.obj_diff_vec_len=30 \
    dataset.data.encoding_type=cached_diffusion_cosin_angle_wocm \
    dataset.data.dataset_directory=bedroom \
    dataset.data.annotation_file=bedroom_threed_front_splits_original.csv \
    dataset.data.room_type=bedroom \
    algorithm.custom.old=True \
    algorithm.ddpo.dynamic_constraint_rewards.reward_base_dir=/home/pramish_paudel/codes/3dhope_rl/dynamic_constraint_rewards \
    algorithm.ddpo.dynamic_constraint_rewards.user_query="Bedroom with tv stand and desk and chair for working." \
    algorithm.ddpo.dynamic_constraint_rewards.agentic=True \
    algorithm.ddpo.dynamic_constraint_rewards.universal_weight=0.0



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
