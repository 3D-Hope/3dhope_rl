#!/bin/bash

# Script to run inference experiments with different configurations
# Experiments for runs: bgdrozky and juy0jvto
# Testing: DDPM/DDIM schedulers with EMA True/False

set -e  # Exit on error

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print experiment header
print_experiment_header() {
    local run_id=$1
    local scheduler=$2
    local ema=$3
    local exp_num=$4
    
    echo ""
    echo "################################################################################"
    echo "################################################################################"
    echo "###"
    echo "###  EXPERIMENT $exp_num"
    echo "###"
    echo "###  Run ID: $run_id"
    echo "###  Scheduler: $scheduler"
    echo "###  EMA: $ema"
    echo "###"
    echo "################################################################################"
    echo "################################################################################"
    echo ""
}

# Function to run experiment
run_experiment() {
    local run_id=$1
    local scheduler=$2
    local ema=$3
    local exp_num=$4
    
    print_experiment_header "$run_id" "$scheduler" "$ema" "$exp_num"
    
    echo "Configuration:"
    echo "  - load: $run_id"
    echo "  - algorithm.noise_schedule.scheduler: $scheduler"
    echo "  - algorithm.ema.use: $ema"
    echo "  - num_scenes: 256"
    echo "  - algorithm.noise_schedule.ddim.num_inference_timesteps: 150"
    echo ""
    echo "Starting experiment at: $(date)"
    echo ""
    
    PYTHONPATH=. python scripts/reward_custom_sample_and_render.py \
        dataset=custom_scene \
        dataset.processed_scene_data_path=data/metadatas/custom_scene_metadata.json \
        dataset.max_num_objects_per_scene=12 \
        +num_scenes=256 \
        algorithm=scene_diffuser_flux_transformer \
        experiment.find_unused_parameters=True \
        algorithm.classifier_free_guidance.use=False \
        algorithm.classifier_free_guidance.weight=0 \
        algorithm.num_additional_tokens_for_sampling=0 \
        algorithm.custom.loss=true \
        algorithm.noise_schedule.ddim.num_inference_timesteps=150 \
        algorithm.trainer=ddpm \
        load=$run_id \
        algorithm.noise_schedule.scheduler=$scheduler \
        algorithm.ema.use=$ema
    
    echo ""
    echo "Experiment completed at: $(date)"
    echo ""
    echo "################################################################################"
    echo "###  END OF EXPERIMENT $exp_num"
    echo "################################################################################"
    echo ""
    echo ""
}

# Main execution
echo ""
echo "================================================================================"
echo "                    INFERENCE EXPERIMENTS BATCH"
echo "================================================================================"
echo ""
echo "Total experiments to run: 8"
echo ""
echo "Run IDs:"
echo "  1. bgdrozky"
echo "  2. juy0jvto"
echo ""
echo "Schedulers: DDPM, DDIM"
echo "EMA settings: True, False"
echo ""
echo "Starting batch at: $(date)"
echo "================================================================================"
echo ""

exp_counter=1

# Experiments for run bgdrozky
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>>"
echo ">>>  STARTING EXPERIMENTS FOR RUN: bgdrozky"
echo ">>>"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ""

run_experiment "bgdrozky" "ddpm" "True" "$exp_counter"
((exp_counter++))

run_experiment "bgdrozky" "ddpm" "False" "$exp_counter"
((exp_counter++))

run_experiment "bgdrozky" "ddim" "True" "$exp_counter"
((exp_counter++))

run_experiment "bgdrozky" "ddim" "False" "$exp_counter"
((exp_counter++))

# Experiments for run juy0jvto
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>>"
echo ">>>  STARTING EXPERIMENTS FOR RUN: juy0jvto"
echo ">>>"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ""

run_experiment "juy0jvto" "ddpm" "True" "$exp_counter"
((exp_counter++))

run_experiment "juy0jvto" "ddpm" "False" "$exp_counter"
((exp_counter++))

run_experiment "juy0jvto" "ddim" "True" "$exp_counter"
((exp_counter++))

run_experiment "juy0jvto" "ddim" "False" "$exp_counter"
((exp_counter++))

# Summary
echo ""
echo "================================================================================"
echo "                    ALL EXPERIMENTS COMPLETED"
echo "================================================================================"
echo ""
echo "Total experiments run: 8"
echo "Completed at: $(date)"
echo ""
echo "Summary of experiments:"
echo ""
echo "Run: bgdrozky"
echo "  1. DDPM + EMA=True"
echo "  2. DDPM + EMA=False"
echo "  3. DDIM + EMA=True"
echo "  4. DDIM + EMA=False"
echo ""
echo "Run: juy0jvto"
echo "  5. DDPM + EMA=True"
echo "  6. DDPM + EMA=False"
echo "  7. DDIM + EMA=True"
echo "  8. DDIM + EMA=False"
echo ""
echo "================================================================================"
echo ""
