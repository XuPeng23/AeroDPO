#!/bin/bash

CUDNN_PATH=$(python -c 'import os; import nvidia.cudnn; print(os.path.dirname(nvidia.cudnn.__file__) + "/lib")')
CUSPARSE_PATH=$(python -c 'import os; import nvidia.cusparse; print(os.path.dirname(nvidia.cusparse.__file__) + "/lib")')
CUBLAS_PATH=$(python -c 'import os; import nvidia.cublas; print(os.path.dirname(nvidia.cublas.__file__) + "/lib")')
TORCH_LIB_PATH=$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__) + "/lib")')
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TORCH_LIB_PATH:$CUDNN_PATH:$CUSPARSE_PATH:$CUBLAS_PATH

export HF_ENDPOINT=https://hf-mirror.com

# Configuration ------------------------------------
PROJECT_ROOT="."

BASE_MODEL_DIR="./model_zoo/AeroVLA-2B-384-Merged"
LORA_DIR="$PROJECT_ROOT/model_zoo/AeroDPO-2B-384-LoRA"


EXP_NAME=$(basename "$LORA_DIR")

GPU_ID=0
INSTANCE_ID=0
PORT=$((30000 + INSTANCE_ID * 5000))

declare -a TASK_LIST=(
    # "seen_valset/Carla_Town01"
    # "seen_valset/Carla_Town04" 
    # "seen_valset/Carla_Town05"
    # "seen_valset/Carla_Town06"
    # "seen_valset/Carla_Town07"
    # "seen_valset/Carla_Town10HD"
    # "seen_valset/Carla_Town15"  

    # "seen_valset/ModernCityMap" 
    # "seen_valset/NewYorkCity"
    # "seen_valset/NYCEnvironmentMegapa"
    # "seen_valset/TropicalIsland"
    
    # "unseen_object_valset/Carla_Town04"
    # "unseen_object_valset/Carla_Town05"
    # "unseen_object_valset/Carla_Town06"
    # "unseen_object_valset/Carla_Town07"
    # "unseen_object_valset/Carla_Town15"
    # "unseen_object_valset/NewYorkCity"
    # "unseen_object_valset/NYCEnvironmentMegapa"
    # "unseen_object_valset/TropicalIsland"

    "unseen_map_valset/ModularPark"
)


echo "Starting batch evaluation..."

for TASK_ID in "${TASK_LIST[@]}"; do
    CATEGORY=$(echo "$TASK_ID" | cut -d'/' -f1)
    MAP_NAME=$(echo "$TASK_ID" | cut -d'/' -f2)
    SAVE_DIR="$PROJECT_ROOT/eval_results/${EXP_NAME}/${CATEGORY}/${MAP_NAME}"
    TEST_JSON="$PROJECT_ROOT/data/uav_dataset/${CATEGORY}_splits/${MAP_NAME}.json"

    if [ -f "$SAVE_DIR/MAP_DONE.txt" ]; then
        echo "[SKIP] $TASK_ID already completed, skipping..."
        continue
    fi

    echo "========================================================"
    echo "Running: $TASK_ID"
    echo "Save Dir: $SAVE_DIR"
    echo "========================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -u $PROJECT_ROOT/src/vlnce_src/eval_aerovla.py \
        --run_type eval \
        --name AeroVLA_Eval \
        --gpu_id $GPU_ID \
        --simulator_tool_port $PORT \
        --DDP_MASTER_PORT 80005 \
        --batchSize 1 \
        --maxWaypoints 200 \
        --dataset_path $PROJECT_ROOT/dataset_raw/ \
        --eval_save_path $SAVE_DIR \
        --model_path $LORA_DIR \
        --base_model_path $BASE_MODEL_DIR \
        --eval_json_path $TEST_JSON \
        --map_spawn_area_json_path $PROJECT_ROOT/data/meta/map_spawnarea_info.json \
        --object_name_json_path $PROJECT_ROOT/data/meta/object_description.json

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        touch "$SAVE_DIR/MAP_DONE.txt"
        echo "Completed: $TASK_ID"
    else
        echo "[ERROR] $TASK_ID failed (exit code: $EXIT_CODE), continuing..."
    fi 
        
    sleep 5
done

echo "All map evaluation tasks finished."
