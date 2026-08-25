#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs_server"
mkdir -p "$LOG_DIR"

echo "Project root: $SCRIPT_DIR"
echo "Logs: $LOG_DIR"

echo "---------------------------------------"
echo "Cleaning up old processes..."
pkill -9 -f "AirVLNSimulatorServerTool.py"
pkill -9 -f "CarlaUE4"
pkill -9 -f "NYCEnvironment"
pkill -9 -f "ModernCity"
pkill -9 -f "LinuxNoEditor"

sleep 3
echo "Cleanup done, starting servers..."
echo "---------------------------------------"

echo "Starting Server 0 -> GPU 0 | port 30000"
CUDA_VISIBLE_DEVICES=0 nohup python -u AirVLNSimulatorServerTool.py \
    --gpus "0" \
    --port 30000 \
    > "$LOG_DIR/server_30000.log" 2>&1 &
sleep 5  # stagger startup to avoid launch contention

echo "Starting Server 1 -> GPU 1 | port 35000"
CUDA_VISIBLE_DEVICES=1 nohup python -u AirVLNSimulatorServerTool.py \
    --gpus "0" \
    --port 35000 \
    > "$LOG_DIR/server_35000.log" 2>&1 &
sleep 5

echo "Starting Server 2 -> GPU 2 | port 40000"
CUDA_VISIBLE_DEVICES=2 nohup python -u AirVLNSimulatorServerTool.py \
    --gpus "0" \
    --port 40000 \
    > "$LOG_DIR/server_40000.log" 2>&1 &
sleep 5

echo "Starting Server 3 -> GPU 3 | port 45000"
CUDA_VISIBLE_DEVICES=3 nohup python -u AirVLNSimulatorServerTool.py \
    --gpus "0" \
    --port 45000 \
    > "$LOG_DIR/server_45000.log" 2>&1 &

echo "---------------------------------------"
echo "All background servers started."
