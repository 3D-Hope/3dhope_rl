#!/bin/bash

# Batch evaluation script for multiple pickle files
# Phase 1: Render all pickle files first
# Phase 2: Run all evaluations for each pickle file
cd /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/ThreedFront
# Define the base directory
BASE_DIR="/media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/"

# Array of pickle files to evaluate with floor conditioning info
# Format: "pkl_file|use_floor"
PKL_FILES_WITH_FLAGS=(
    "$BASE_DIR/outputs/2025-10-18/08-27-02/sampled_scenes_results.pkl|no_floor"   # 1. Flux Transformer RL, qbyilta9, ddpm, True, 1000
    "$BASE_DIR/outputs/2025-10-18/08-31-13/sampled_scenes_results.pkl|no_floor"   # 2. Flux Transformer RL, qbyilta9, ddim, True, 150
    "$BASE_DIR/outputs/2025-10-18/08-32-27/sampled_scenes_results.pkl|no_floor"   # 3. Continuous MiDiffusion RL, qhns5khl, ddpm, True, 1000
    "$BASE_DIR/outputs/2025-10-18/08-36-07/sampled_scenes_results.pkl|no_floor"   # 4. Continuous MiDiffusion RL, qhns5khl, ddim, True, 150
    "$BASE_DIR/outputs/2025-10-18/08-37-12/sampled_scenes_results.pkl|with_floor" # 5. Continuous MiDiffusion Floor, rrudae6n, ddpm, True, 1000
    "$BASE_DIR/outputs/2025-10-18/08-41-48/sampled_scenes_results.pkl|with_floor" # 6. Continuous MiDiffusion Floor, rrudae6n, ddim, True, 150
)

# Get total number of files
TOTAL=${#PKL_FILES_WITH_FLAGS[@]}

echo "==============================================="
echo "Batch Evaluation Script (2-Phase)"
echo "Total files to process: $TOTAL"
echo "==============================================="

# Log file for summary
LOG_FILE="batch_eval_$(date +%Y%m%d_%H%M%S).log"
echo "Batch Evaluation Log - $(date)" > "$LOG_FILE"
echo "======================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Activate conda environment
echo ""
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate midiffusion

# ========================================
# PARALLEL PROCESSING: Render + Evaluate each PKL in its own job
# ========================================

# Concurrency: number of parallel jobs. Override via env var CONCURRENCY.
CONCURRENCY=${CONCURRENCY:-1}

# Create a FIFO semaphore to limit concurrency
FIFO="/tmp/batch_eval_fifo.$$"
mkfifo "$FIFO"
exec 3<> "$FIFO"
rm "$FIFO"
# Seed tokens
for i in $(seq 1 $CONCURRENCY); do
    echo >&3
done

process_entry() {
    local PKL_FILE="$1"
    local FLOOR_FLAG="$2"
    local IDX="$3"

    echo "" | tee -a "$LOG_FILE"
    echo "-----------------------------------------------" | tee -a "$LOG_FILE"
    echo "[${IDX}] START processing: $(basename $(dirname "$PKL_FILE")) [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"

    if [ ! -f "$PKL_FILE" ]; then
        echo "SKIPPED (not found): $PKL_FILE" | tee -a "$LOG_FILE"
        echo >&3
        return
    fi

    # # -------- Render --------
    # START_TIME_RENDER=$(date +%s)
    # if [ "$FLOOR_FLAG" = "with_floor" ]; then
    #     RENDER_CMD=(python scripts/render_results.py "$PKL_FILE" --no_texture)
    # else
    #     RENDER_CMD=(python scripts/render_results.py "$PKL_FILE" --no_texture --without_floor)
    # fi

    # echo "[${IDX}] Running render: ${RENDER_CMD[*]}" | tee -a "$LOG_FILE"
    # if "${RENDER_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    #     END_TIME_RENDER=$(date +%s)
    #     DURATION_RENDER=$((END_TIME_RENDER - START_TIME_RENDER))
    #     echo "[${IDX}] ✅ Rendered in ${DURATION_RENDER}s [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"
    # else
    #     echo "[${IDX}] ❌ Rendering FAILED for $PKL_FILE" | tee -a "$LOG_FILE"
    #     echo >&3
    #     return
    # fi

    # -------- Evaluations --------
    START_TIME_EVAL=$(date +%s)
    EVAL_SUCCESS=true
    if [ "$FLOOR_FLAG" = "with_floor" ]; then
        FLOOR_EVAL_FLAG=""
    else
        FLOOR_EVAL_FLAG="--no_floor"
    fi

    echo "[${IDX}] [1/6] Computing FID scores... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"
    if ! python scripts/compute_fid_scores.py "$PKL_FILE" \
        --output_directory ./fid_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ \
        $FLOOR_EVAL_FLAG 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi

    echo "[${IDX}] [2/6] Computing KID scores... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"
    if ! python scripts/compute_fid_scores.py "$PKL_FILE" \
        --compute_kid \
        --output_directory ./fid_tmps \
        --no_texture \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ \
        $FLOOR_EVAL_FLAG 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi

    echo "[${IDX}] [3/6] Running bbox analysis..." | tee -a "$LOG_FILE"
    if ! python scripts/bbox_analysis.py "$PKL_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi

    echo "[${IDX}] [4/6] Computing KL divergence..." | tee -a "$LOG_FILE"
    if ! python scripts/evaluate_kl_divergence_object_category.py "$PKL_FILE" \
        --output_directory ./kl_tmps 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi

    echo "[${IDX}] [5/6] Calculating object count..." | tee -a "$LOG_FILE"
    if ! python scripts/calculate_num_obj.py "$PKL_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi

    echo "[${IDX}] [6/6] Running classifier... [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"
    if ! python scripts/synthetic_vs_real_classifier.py "$PKL_FILE" \
        --output_directory ./classifier_tmps \
        --no_texture \
        $FLOOR_EVAL_FLAG \
        --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ 2>&1 | tee -a "$LOG_FILE"; then
        EVAL_SUCCESS=false
    fi

    END_TIME_EVAL=$(date +%s)
    DURATION_EVAL=$((END_TIME_EVAL - START_TIME_EVAL))

    if [ "$EVAL_SUCCESS" = true ]; then
        echo "[${IDX}] ✅ SUCCESS: All evaluations completed in ${DURATION_EVAL}s - $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"
        echo "SUCCESS: $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}] - Duration: ${DURATION_EVAL}s" >> "$LOG_FILE"
    else
        echo "[${IDX}] ❌ FAILED: Some evaluations failed - $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}]" | tee -a "$LOG_FILE"
        echo "FAILED: $(basename $(dirname $PKL_FILE)) [${FLOOR_FLAG}] - Duration: ${DURATION_EVAL}s" >> "$LOG_FILE"
    fi

    echo >&3
}

# Launch jobs in parallel (each job will render+evaluate its PKL)
COUNTER=1
IDX=1
for ENTRY in "${PKL_FILES_WITH_FLAGS[@]}"; do
    IFS='|' read -r PKL_FILE FLOOR_FLAG <<< "$ENTRY"
    # Acquire token (blocks if concurrency limit reached)
    read -u 3
    process_entry "$PKL_FILE" "$FLOOR_FLAG" "$IDX" &
    ((IDX++))
done

# Wait for all background jobs to finish
wait

# Close semaphore fd
exec 3>&-

echo ""
echo "==============================================="
echo "Batch Evaluation Complete!"
echo "==============================================="
echo "Processed: $TOTAL files"
echo "Log saved to: $LOG_FILE"
echo ""
echo "Summary:"
echo "--------"
grep -E "SUCCESS|FAILED|SKIPPED|RENDERED" "$LOG_FILE" | grep -v "PHASE"
echo ""
