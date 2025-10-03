#!/bin/bash

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400
export NODE_RANK=$SLURM_NODEID            # The rank of the current node (0, 1, 2, 3)

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

# 4. ALL NODES ARE POLICY NODES (reward model co-located)
echo "=== Node $NODE_RANK: POLICY NODE (with co-located transformers reward model) ==="

# 5. START RAY CLUSTER ON ALL NODES (0-3)
echo "Starting Ray cluster setup on node $NODE_RANK with $GPUS_PER_NODE GPUs. Master is at $MASTER_ADDR:$MASTER_PORT."

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node ==="
    uv run ray start --head --disable-usage-stats --num-gpus=$GPUS_PER_NODE
    echo "Ray head started successfully"

    # Wait for worker nodes to connect
    echo "Waiting for worker nodes to connect..."
    sleep 30

else
    echo "=== Starting Ray WORKER node ==="
    # Wait for head node to be ready
    sleep 15

    uv run ray start --address=$MASTER_ADDR:6379 --disable-usage-stats --num-gpus=$GPUS_PER_NODE
    echo "Ray worker node $NODE_RANK connected successfully"
    sleep 5
fi

# 6. EXECUTE THE GRPO TRAINING JOB (only on head node)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting NeMo RL GRPO Training with Transformers Reward Model ==="

    # Wait for all nodes to join
    echo "Waiting for all nodes to join cluster..."
    sleep 30

    # Check Ray cluster status
    echo "Checking Ray cluster status..."
    uv run ray status

    echo "✓ Using co-located transformers reward model on all nodes"

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
