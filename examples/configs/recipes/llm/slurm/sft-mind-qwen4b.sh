#!/bin/bash
# SLURM batch script for NeMo RL Multi-Node SFT Training
# 8 nodes with 8 GPUs each

#SBATCH --job-name=nemo-rl-sft-mind-qwen4b-3epoch-merge1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/nemo-rl-sft-mind-qwen4b-3epoch-merge1-%j.out
#SBATCH --error=slurm_logs/nemo-rl-sft-mind-qwen4b-3epoch-merge1-%j.err

# Create a script that will be executed on each node via srun (in shared location)
cat > /home/pmartins/nemo-rl/examples/configs/recipes/llm/slurm/slurm_multinode_worker.sh << 'WORKER_SCRIPT_EOF'
#!/bin/bash

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29400
export NODE_RANK=$SLURM_NODEID            # The rank of the current node (0, 1)

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

# --- NCCL DEBUGGING (IMPROVED) ---
# Create a dedicated directory for NCCL logs inside your main slurm_logs
# NCCL_LOG_DIR="/home/pmartins/nemo-rl/slurm_logs/nccl_logs_${SLURM_JOB_ID}"
# mkdir -p "$NCCL_LOG_DIR"

# Set NCCL debug level and specify a UNIQUE, ABSOLUTE path for the log file
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_ALL
# export NCCL_DEBUG_FILE="${NCCL_LOG_DIR}/nccl_debug_node${SLURM_NODEID}_$(hostname -s).log"
# echo "  NCCL Log File: ${NCCL_DEBUG_FILE}"
# --- END NCCL DEBUGGING ---

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

# 4. START RAY CLUSTER
echo "Starting Ray cluster setup on node $NODE_RANK with $GPUS_PER_NODE GPUs. Master is at $MASTER_ADDR:$MASTER_PORT."

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node ==="
    uv run ray start --head --disable-usage-stats
    echo "Ray head started successfully"
    
    # Wait for worker to connect
    echo "Waiting for worker node to connect..."
    sleep 20
    
else
    echo "=== Starting Ray WORKER node ==="
    # Wait for head node to be ready
    sleep 15
    
    uv run ray start --address=$MASTER_ADDR:6379 --disable-usage-stats
    echo "Ray worker node $NODE_RANK connected successfully"
    sleep 5
fi


# 5. EXECUTE THE TRAINING JOB (only on head node)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting NeMo RL SFT Training ==="
    
    # Create a detailed log with timestamps and node info
    LOG_DIR="/home/pmartins/nemo-rl/slurm_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/training_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "📊 Starting training on node $NODE_RANK at $(date)" | tee -a "$LOG_FILE"

    # Run with detailed logging and real-time output
    uv run python examples/run_sft.py \
        --config examples/configs/recipes/llm/sft-mind-megatron-qwen4b.yaml \
        2>&1 | tee -a "$LOG_FILE"

    TRAINING_EXIT_CODE=$?
    
    echo "=== Training completed with exit code $TRAINING_EXIT_CODE ==="
    
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
WORKER_SCRIPT_EOF

# Make the worker script executable
chmod +x /home/pmartins/nemo-rl/examples/configs/recipes/llm/slurm/slurm_multinode_worker.sh

echo "=== Starting Multi-Node SLURM Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"

# Launch the worker script on all nodes simultaneously using srun
srun --ntasks-per-node=1 /home/pmartins/nemo-rl/examples/configs/recipes/llm/slurm/slurm_multinode_worker.sh

echo "=== Multi-Node Job Completed ==="
                                                