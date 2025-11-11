#!/bin/bash
# SLURM batch script for NeMo RL Multi-Node SFT Training with DTensor
# 4 nodes with 8 GPUs each

#SBATCH --job-name=nemo-rl-sft-mind-qwen235ba22b-thinking-mind-expert-dtensor
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=72:00:00
#SBATCH --output=slurm_logs/nemo-rl-sft-mind-qwen235ba22b-thinking-mind-expert-dtensor-%j.out
#SBATCH --error=slurm_logs/nemo-rl-sft-mind-qwen235ba22b-thinking-mind-expert-dtensor-%j.err

# Create a script that will be executed on each node via srun (in shared location)
cat > /home/pmartins/nemo-rl/examples/configs/recipes/llm/slurm/slurm_multinode_worker_qwen235ba22b_thinking-mind-expert-dtensor.sh << 'WORKER_SCRIPT_EOF'
#!/bin/bash

# 1. SET UP DISTRIBUTED ENVIRONMENT VARIABLES FOR SLURM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29401  # Different port to avoid conflicts
export NODE_RANK=$SLURM_NODEID            # The rank of the current node (0, 1, 2, 3)

# This variable holds the number of GPUs per node
GPUS_PER_NODE=8
export GPUS_PER_NODE

echo "Environment check:"
echo "  GPUS_PER_NODE: $GPUS_PER_NODE"
echo "  NODE_RANK: $NODE_RANK"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"

# 2. SET UP PATHS AND ENVIRONMENT
cd /home/pmartins/nemo-rl/

# Ensure uv is available
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# 3. SET APPLICATION-SPECIFIC VARIABLES (DTensor optimized)
export HF_HOME="/mnt/data/shared/cache"
export HF_DATASETS_CACHE="/mnt/data/shared/cache"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

# DTensor/FSDP specific environment variables
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_DISTRIBUTED_DEBUG=INFO  # Enable for DTensor debugging
export FSDP_CPU_RAM_EFFICIENT_LOADING=1  # Enable efficient FSDP loading
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# Force Python to show all output immediately
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# --- NCCL OPTIMIZATIONS FOR DTENSOR ---
export NCCL_P2P_LEVEL=NVL
export NCCL_P2P_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET=IB
export NCCL_SOCKET_IFNAME=eth0

# DTensor/FSDP specific NCCL settings
export NCCL_TREE_THRESHOLD=0                # Use tree algorithm for better scaling

# NCCL Timeout and Reliability Settings (comprehensive)
export NCCL_IB_TIMEOUT=22                    # InfiniBand timeout (default: 20)
export NCCL_IB_RETRY_CNT=7                   # Retry count (default: 7)
export NCCL_TIMEOUT_US=1800000000           # 30 minutes in microseconds
export NCCL_ASYNC_ERROR_HANDLING=1          # Enable async error handling
export NCCL_BLOCKING_WAIT=1                 # Use blocking waits
export NCCL_CUMEM_ENABLE=0                  # Disable CUDA memory pools for stability

# PyTorch Distributed Timeout Settings
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT=1800  # 30 minutes in seconds
export TORCH_NCCL_BLOCKING_WAIT=1               # Use blocking waits in PyTorch
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1        # Enable async error handling in PyTorch

# TCP Store Timeout Settings (for DTensor initialization) - Extended for 235B model
export TORCH_TCP_STORE_TIMEOUT=3600             # 1 hour for TCP store operations
export TORCH_DISTRIBUTED_TIMEOUT_MINUTES=60    # 1 hour for distributed initialization

HOSTNAME_SHORT=$(hostname -s)
export RAY_TMPDIR="/tmp/ray/ray_dtensor_${USER}_${SLURM_JOB_ID}_${NODE_RANK}"
mkdir -p "$RAY_TMPDIR"
echo "  RAY_TMPDIR: $RAY_TMPDIR (node-specific, isolated)"
export TMPDIR="$RAY_TMPDIR"

# Ray timeouts (more generous for DTensor)
export RAY_START_TIMEOUT_SECONDS=600  # 10 minutes for DTensor initialization
export RAY_gcs_server_request_timeout_seconds=180
export RAY_raylet_heartbeat_timeout_milliseconds=120000  # 2 minutes
export RAY_num_heartbeats_timeout=60
export RAY_raylet_client_num_connect_attempts=30
export RAY_gcs_rpc_server_reconnect_timeout_s=180

# Ray cluster formation timeouts
export RAY_TIMEOUT_MS=600000  # 10 minutes
export RAY_REDIS_START_RETRIES=30

# 4. START RAY CLUSTER
echo "Starting Ray cluster setup for DTensor on node $NODE_RANK with $GPUS_PER_NODE GPUs. Master is at $MASTER_ADDR:$MASTER_PORT."

if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting Ray HEAD node for DTensor ==="
    uv run ray start --head --disable-usage-stats --port=6380  # Different port
    echo "Ray head started successfully"

    # Wait longer for worker nodes (DTensor initialization can be slow)
    echo "Waiting for worker nodes to connect..."
    sleep 45

else
    echo "=== Starting Ray WORKER node for DTensor ==="
    # Wait longer for head node to be ready
    sleep 30

    uv run ray start --address=$MASTER_ADDR:6380 --disable-usage-stats
    echo "Ray worker node $NODE_RANK connected successfully"
    sleep 10
fi


# 5. EXECUTE THE TRAINING JOB (only on head node)
if [ "$NODE_RANK" -eq 0 ]; then
    echo "=== Starting NeMo RL SFT Training with DTensor ==="

    # Create a detailed log with timestamps and node info
    LOG_DIR="/home/pmartins/nemo-rl/slurm_logs/$(date +%Y%m%d)"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/training_qwen235ba22b_thinking_mind_expert_dtensor_node_${NODE_RANK}_$(date +%H%M%S).log"

    echo "📊 Starting DTensor training on node $NODE_RANK at $(date)" | tee -a "$LOG_FILE"
    echo "📊 Using DTensor config with FSDP enabled" | tee -a "$LOG_FILE"

    # Run with detailed logging and real-time output
    uv run python examples/run_sft.py \
        --config examples/configs/recipes/llm/sft-mind-qwen235ba22b-thinking-mind-expert-dtensor.yaml \
        2>&1 | tee -a "$LOG_FILE"

    TRAINING_EXIT_CODE=$?

    echo "=== DTensor training completed with exit code $TRAINING_EXIT_CODE ==="

    # Shutdown Ray cluster
    echo "=== Shutting down Ray cluster ==="
    uv run ray stop

    exit $TRAINING_EXIT_CODE
else
    echo "=== Worker node $NODE_RANK waiting for DTensor training to complete ==="

    # Worker nodes wait for the training to complete
    while uv run ray status > /dev/null 2>&1; do
        sleep 30
    done

    echo "=== Worker node $NODE_RANK shutting down ==="
    uv run ray stop
fi
WORKER_SCRIPT_EOF

# Make the worker script executable
chmod +x /home/pmartins/nemo-rl/examples/configs/recipes/llm/slurm/slurm_multinode_worker_qwen235ba22b_thinking-mind-expert-dtensor.sh

echo "=== Starting Multi-Node DTensor SLURM Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Using DTensor with FSDP for memory efficiency"

# Launch the worker script on all nodes simultaneously using srun
srun --ntasks-per-node=1 /home/pmartins/nemo-rl/examples/configs/recipes/llm/slurm/slurm_multinode_worker_qwen235ba22b_thinking-mind-expert-dtensor.sh

echo "=== Multi-Node DTensor Job Completed ==="