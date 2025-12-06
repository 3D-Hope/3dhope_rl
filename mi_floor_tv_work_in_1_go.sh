#!/bin/bash
#SBATCH --job-name=mi_floor_tv_work_in_1_go
#SBATCH --partition=batch
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=20G
#SBATCH --time=15:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Exit on error but with better error reporting
set -euo pipefail
trap 'ERR_CODE=$?; echo "❌ Error on line $LINENO. Exit code: $ERR_CODE" >&2; exit $ERR_CODE' ERR
trap 'echo "🛑 Job interrupted"; exit 130' INT

# -------------------------
# Basic setup / logging
# -------------------------
mkdir -p logs
echo "──────────────────────────────────────────────────────────────────────────────"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "──────────────────────────────────────────────────────────────────────────────"
echo ""

echo "System information:"
free -h || true
df -h /scratch/pramish_paudel || df -h . || true
echo ""

export WANDB_ENTITY="078bct021-ashok-d"

# -------------------------
# Stage 1 & 2: copy caches/dataset (keeps your original behavior)
# -------------------------
# (kept your rsync/unzip logic but robustified)
echo "STAGE 1: Copy/extract caches and dataset..."

rm -rf /scratch/pramish_paudel/bedroom_sdf_cache || true
if [ ! -d "/scratch/pramish_paudel/bedroom_sdf_cache" ]; then
    echo "Copying SDF cache..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/bedroom_sdf_cache.zip /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy SDF cache"; exit 1
    }
    echo "Extracting SDF cache..."
    unzip -o /scratch/pramish_paudel/bedroom_sdf_cache.zip -d /scratch/pramish_paudel/ || {
        echo "❌ Failed to extract SDF cache"; exit 1
    }
    rm -f /scratch/pramish_paudel/bedroom_sdf_cache.zip
    echo "✅ SDF cache copied"
else
    echo "✅ SDF cache already exists in scratch"
fi
ls -la /scratch/pramish_paudel/bedroom_sdf_cache || true

rm -rf /scratch/pramish_paudel/bedroom_accessibility_cache || true
if [ ! -d "/scratch/pramish_paudel/bedroom_accessibility_cache" ]; then
    echo "Copying accessibility cache..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/bedroom_accessibility_cache.zip /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy accessibility cache"; exit 1
    }
    unzip -o /scratch/pramish_paudel/bedroom_accessibility_cache.zip -d /scratch/pramish_paudel/ || {
        echo "❌ Failed to extract accessibility cache"; exit 1
    }
    rm -f /scratch/pramish_paudel/bedroom_accessibility_cache.zip
    echo "✅ accessibility cache copied"
else
    echo "✅ accessibility cache already exists in scratch"
fi
ls -la /scratch/pramish_paudel/bedroom_accessibility_cache || true

echo "STAGE 2: Checking bedroom dataset..."
if [ ! -d "/scratch/pramish_paudel/bedroom" ]; then
    echo "Copying bedroom dataset..."
    rsync -aHzv --progress /home/pramish_paudel/3dhope_data/bedroom.zip /scratch/pramish_paudel/ || {
        echo "❌ Failed to copy bedroom dataset"; exit 1
    }
    echo "Extracting dataset..."
    cd /scratch/pramish_paudel/
    unzip -oq bedroom.zip || { echo "❌ Failed to extract bedroom dataset"; exit 1; }
    rm -f bedroom.zip
    echo "✅ bedroom dataset extracted"
else
    echo "✅ bedroom dataset already exists in scratch"
fi
echo ""

# -------------------------
# Stage 3: Miniforge/Conda setup
# -------------------------
echo "STAGE 3: Setting up Miniforge (if missing)..."
CONDA_DIR="/scratch/pramish_paudel/tools/miniforge"
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniforge to $CONDA_DIR..."
    mkdir -p /scratch/pramish_paudel/tools/
    cd /scratch/pramish_paudel/tools/
    # Use uname to choose correct installer
    MINIFORGE_SH="Miniforge3-$(uname)-$(uname -m).sh"
    wget -q --show-progress "https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE_SH" -O miniforge.sh || {
        echo "❌ Failed to download Miniforge"; exit 1
    }
    bash miniforge.sh -b -p "$CONDA_DIR" || { echo "❌ Failed to install Miniforge"; exit 1; }
    rm -f miniforge.sh
    echo "✅ Miniforge installed at $CONDA_DIR"
else
    echo "✅ Miniforge already exists at $CONDA_DIR"
fi

# Source conda hooks
echo "Sourcing conda..."
# shellcheck source=/dev/null
source "$CONDA_DIR/etc/profile.d/conda.sh" || { echo "❌ Failed to source conda.sh"; exit 1; }
# ensure conda command is available
eval "$($CONDA_DIR/bin/conda shell.bash hook)" || true

echo ""

# -------------------------
# Stage 4: Create and activate conda env
# -------------------------
echo "STAGE 4: Creating/activating conda env '3dhope_rl'..."
CONDA_ENV_NAME="3dhope_rl"
if ! conda env list | awk '{print $1}' | grep -xq "$CONDA_ENV_NAME"; then
    echo "Creating conda environment: $CONDA_ENV_NAME (python 3.10)"
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y || { echo "❌ Failed to create conda env"; exit 1; }
fi

# Activate env
echo "Activating conda environment: $CONDA_ENV_NAME"
conda activate "$CONDA_ENV_NAME" || { echo "❌ Failed to activate conda env"; exit 1; }

# Make sure conda's bin is first in PATH, so conda-installed poetry is used
export PATH="$CONDA_PREFIX/bin:$PATH"
hash -r || true

echo "Environment verification:"
echo "  Active conda environment: ${CONDA_DEFAULT_ENV:-N/A}"
echo "  Python path: $(which python)"
echo "  Python version: $(python --version 2>&1)"
echo "  Pip path: $(which pip)"
echo ""

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


# Run poetry install -- prefer no interaction. If it fails, fallback to pip editable install.
if "$POETRY_CMD" install --no-interaction --no-ansi 2>&1 | tee "$POETRY_INSTALL_LOG"; then
    echo "✅ Poetry install succeeded"
else
    echo "⚠️ Poetry install failed — showing last 30 lines of log:"
    tail -n 30 "$POETRY_INSTALL_LOG" || true
    echo "Falling back to pip install -e . (best-effort)"
    pip install -e . || { echo "❌ Fallback pip install failed"; exit 1; }
fi

# -------------------------
# Stage 8: Activate project's .venv (if created) OR fall back to conda env
# -------------------------
if [ -d ".venv" ]; then
    echo "Activating project .venv..."
    # shellcheck disable=SC1091
    source .venv/bin/activate || { echo "❌ Failed to activate .venv"; exit 1; }
    echo "✅ Using Poetry .venv: $(which python)"
else
    echo "⚠️ No .venv found — will continue using conda env: ${CONDA_DEFAULT_ENV:-N/A}"
fi

# quick verify that hydra (or other crucial libs) are importable
echo "Verifying important packages (hydra)..."
python - <<'PYTEST' || { echo "❌ Required python imports failed"; exit 1; }
try:
    import importlib, sys
    modnames = ["hydra", "omegaconf"]
    missing = []
    for m in modnames:
        try:
            importlib.import_module(m)
        except Exception as e:
            missing.append((m,str(e)))
    if missing:
        print("MISSING:", missing)
        sys.exit(2)
    else:
        print("All checks passed:", [importlib.import_module(m).__name__ for m in modnames])
except Exception as e:
    print("Import-time error:", str(e))
    raise
PYTEST

echo ""

# -------------------------
# Stage 9: GPU check
# -------------------------
echo "STAGE 9: GPU check (nvidia-smi):"
nvidia-smi || echo "⚠️ nvidia-smi failed or not present on this node"

# -------------------------
# Stage 10: Run training
# -------------------------
echo "✅ All dependencies installed and configured"
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "STAGE 10: Starting RL training..."
echo "Training started at: $(date)"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

export PYTHONUNBUFFERED=1
export DISPLAY=:0

# ---- your large python command (kept original flags) ----
PYTHONPATH=. python -u main.py +name=mi_floor \
    load=rrudae6n  \
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
    algorithm.classifier_free_guidance.use_floor=true \
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

# -------------------------
# Final status
# -------------------------
EXIT_CODE=$?
echo ""
echo "════════════════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully at: $(date)"
    exit 0
else
    echo "❌ Training failed with exit code $EXIT_CODE at: $(date)"
    exit $EXIT_CODE
fi
echo "════════════════════════════════════════════════════════════════════════"
