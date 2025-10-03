#!/bin/bash

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400
export NODE_RANK=$SLURM_NODEID            # The rank of the current node (0, 1, 2, ..., 7)

# This variable holds the number of GPUs per node
GPUS_PER_NODE=8
export GPUS_PER_NODE

echo "Environment check:"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"

# 2. SET UP PATHS AND ENVIRONMENT
cd /home/pmartins/nemo-rl/

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 3. SET APPLICATION-SPECIFIC VARIABLES (with fix for cache)
export HF_HOME="/mnt/data/shared/cache" # Use HF_HOME instead of the deprecated TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="/mnt/data/shared/cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ENABLE_MONITORING=0

# Force Python to show all output immediately
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Add Megatron-Bridge to Python path
export PYTHONPATH="/home/pmartins/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:$PYTHONPATH"

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=eth0

HOSTNAME_SHORT=$(hostname -s)
export RAY_TMPDIR="/tmp/ray/ray_${USER}_${SLURM_JOB_ID}_${NODE_RANK}"
mkdir -p "$RAY_TMPDIR"
echo "  RAY_TMPDIR: $RAY_TMPDIR (node-specific, isolated)"
export TMPDIR="$RAY_TMPDIR"
export RAY_START_TIMEOUT_SECONDS=300  # 5 minutes instead of 30 seconds
export RAY_gcs_server_request_timeout_seconds=120
export RAY_raylet_heartbeat_timeout_milliseconds=90000  # 90 seconds
export RAY_num_heartbeats_timeout=50
export RAY_raylet_client_num_connect_attempts=20
export RAY_gcs_rpc_server_reconnect_timeout_s=120

# Ray cluster formation timeouts
export RAY_TIMEOUT_MS=300000  # 5 minutes
export RAY_REDIS_START_RETRIES=20

# 4. DETERMINE NODE ROLE (Policy nodes: 0-3, Reward model node: 4)
if [ "$NODE_RANK" -lt 4 ]; then
    NODE_TYPE="policy"
    echo "=== Node $NODE_RANK: POLICY NODE ==="
else
    NODE_TYPE="reward"
    echo "=== Node $NODE_RANK: REWARD MODEL NODE ==="
fi

# 5. START REWARD MODEL SERVER ON NODE 4
if [ "$NODE_TYPE" = "reward" ]; then
    echo "=== Starting VLLM reward model server on node $NODE_RANK ==="

    # Extract reward model path from config file
    REWARD_MODEL_PATH=$(grep "reward_model_name:" /home/pmartins/nemo-rl/examples/configs/recipes/llm/grpo-megatron-qwen4b-reward-model.yaml | head -1 | cut -d'"' -f2)
    echo "Using reward model: $REWARD_MODEL_PATH"

    # Server configuration
    REWARD_PORT=8000
    TP_SIZE=4  # Use 4 GPUs for 30B model
    echo "Starting VLLM server on port $REWARD_PORT with tensor parallelism $TP_SIZE"

    # Create log files for VLLM output
    VLLM_LOG_DIR="/home/pmartins/nemo-rl/grpo_slurm_logs/vllm_logs"
    mkdir -p "$VLLM_LOG_DIR"
    VLLM_STDOUT_LOG="$VLLM_LOG_DIR/vllm_stdout_node${NODE_RANK}.log"
    VLLM_STDERR_LOG="$VLLM_LOG_DIR/vllm_stderr_node${NODE_RANK}.log"

    # Start VLLM server for reward model - remove classify task, use standard chat
    uv run vllm serve "$REWARD_MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$REWARD_PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization 0.9 \
        --max-model-len 4096 \
        --served-model-name "reward-model" \
        --disable-log-stats \
        --disable-frontend-multiprocessing \
        > "$VLLM_STDOUT_LOG" 2> "$VLLM_STDERR_LOG" &

    VLLM_PID=$!
    echo "VLLM server started with PID $VLLM_PID"

    # Wait for startup with minimal feedback
    echo "Waiting for VLLM server to be ready..."
    for i in {1..24}; do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "ERROR: VLLM process died during startup!"
            echo "Check logs: $VLLM_STDOUT_LOG and $VLLM_STDERR_LOG"
            tail -10 "$VLLM_STDERR_LOG" 2>/dev/null
            exit 1
        fi

        # Check if server is responding
        if curl -s "http://localhost:$REWARD_PORT/health" > /dev/null 2>&1; then
            echo "✓ VLLM server is ready and responding"
            break
        fi

        if [ $((i % 6)) -eq 0 ]; then
            echo "Still starting... (${i}/24)"
        fi
        sleep 5
    done

    # Final startup check
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: VLLM failed to start properly"
        tail -20 "$VLLM_STDERR_LOG" 2>/dev/null
        exit 1
    fi

    echo "✓ VLLM reward model server running successfully"

    # Keep server running with minimal logging
    while kill -0 $VLLM_PID 2>/dev/null; do
        sleep 60  # Check every minute instead of 30 seconds
    done

    echo "ERROR: VLLM reward model server stopped unexpectedly!"
    echo "Last error logs:"
    tail -20 "$VLLM_STDERR_LOG" 2>/dev/null
    exit 1
fi

# 6. START RAY CLUSTER ON POLICY NODES (0-3)
echo "Starting Ray cluster setup on policy node $NODE_RANK with $GPUS_PER_NODE GPUs. Master is at $MASTER_ADDR:$MASTER_PORT."

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node ==="
    uv run ray start --head --disable-usage-stats
    echo "Ray head started successfully"

    # Wait for worker nodes to connect
    echo "Waiting for worker nodes to connect..."
    sleep 30

else
    echo "=== Starting Ray WORKER node ==="
    # Wait for head node to be ready
    sleep 15

    uv run ray start --address=$MASTER_ADDR:6379 --disable-usage-stats
    echo "Ray worker node $NODE_RANK connected successfully"
    sleep 5
fi

# 7. EXECUTE THE GRPO TRAINING JOB (only on head node)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting NeMo RL GRPO Training with VLLM Reward Model ==="

    # Wait for reward model servers to be ready
    echo "Waiting for reward model servers to be ready..."
    sleep 90

    # Test reward model connectivity
    REWARD_NODE_4=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n '5p')
    echo "Testing connection to reward model at $REWARD_NODE_4:8000"

    # Update config with actual reward model URL
    sed -i "s|reward_model_url: \"http://PLACEHOLDER_REWARD_NODE:8000\"|reward_model_url: \"http://$REWARD_NODE_4:8000\"|g" \
        examples/configs/recipes/llm/grpo-megatron-qwen4b-reward-model.yaml

    echo "Updated reward model URL to: http://$REWARD_NODE_4:8000"

    # Create a detailed log with timestamps and node info
    LOG_DIR="/home/pmartins/nemo-rl/grpo_slurm_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/grpo_training_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "📊 Starting GRPO training on node $NODE_RANK at $(date)" | tee -a "$LOG_FILE"

    # Run GRPO training with detailed logging and real-time output
    uv run python examples/run_grpo.py \
        --config examples/configs/recipes/llm/grpo-megatron-qwen4b-reward-model.yaml \
        2>&1 | tee -a "$LOG_FILE"

    TRAINING_EXIT_CODE=$?

    echo "=== GRPO Training completed with exit code $TRAINING_EXIT_CODE ==="

    # Shutdown Ray cluster
    echo "=== Shutting down Ray cluster ==="
    uv run ray stop

    exit $TRAINING_EXIT_CODE
else
    echo "=== Worker node $NODE_RANK waiting for training to complete ==="

    # Worker nodes wait for the training to complete
    while uv run ray status > /dev/null 2>&1; do
        sleep 30
    done

    echo "=== Worker node $NODE_RANK shutting down ==="
    uv run ray stop
fi
