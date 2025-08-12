#!/bin/bash

# SBATCH --job-name=interactive-notebook
# SBATCH --output=$LOG/notebooks/notebook_job_%j.log
# SBATCH --account=$ACCOUNT
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=16
# SBATCH --mem=64G
# SBATCH --gpus=1
# SBATCH --partition=interactive
# SBATCH --container-image=/lustre/fsw/portfolios/llmservice/users/mfathi/containers/nemo_rl_base.sqsh
# SBATCH --time=00:30:00

VENV_DIR=".venv"
KERNEL_NAME="slurm-job-kernel-mfathi"

echo "===================================================================="
echo "Starting SLURM job for interactive Jupyter Notebook"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Log file: $LOG/notebooks/notebook_job_${SLURM_JOB_ID}.log"
echo "Virtual Environment: $(pwd)/${VENV_DIR}"
echo "===================================================================="

# Step 1: Set up Python virtual environment with uv
echo
echo "[1/4] Setting up Python virtual environment using uv..."
if [ -d "$VENV_DIR" ]; then
    echo "Existing virtual environment found. Reusing."
else
    echo "Creating new virtual environment..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi
source $VENV_DIR/bin/activate
echo "Virtual environment activated."
echo

# Step 2: Install dependencies from requirements.txt using uv
echo "[2/4] Installing Python dependencies using uv..."
uv pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies. Check requirements.txt and uv setup."
    exit 1
fi
echo "Dependencies installed successfully."
echo

# Step 3: Register the virtual environment as a Jupyter kernel
echo "[3/4] Registering virtual environment as a Jupyter kernel..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="SLURM Job Kernel"
if [ $? -ne 0 ]; then
    echo "Error: Failed to register Jupyter kernel."
    exit 1
fi
echo "Jupyter kernel '$KERNEL_NAME' registered."
echo

# Step 4: Start Jupyter Lab server on a random port
echo "[4/4] Starting Jupyter Lab server..."
# Find a random available port
PORT=$(shuf -i 8000-9999 -n 1)
IP_ADDRESS=$(hostname -I | awk '{print $1}')

# Start jupyter lab in the background
jupyter lab --no-browser --port=${PORT} --ip=0.0.0.0 &

# Wait a few seconds for the server to start up
sleep 15

echo "Jupyter Lab server is starting in the background."
echo

# Step 5: Display connection instructions
echo
echo "==================== CONNECTION INSTRUCTIONS ===================="
echo
echo "----------[ LOCAL TERMINAL ]----------"
echo "1. Open a NEW terminal on your LOCAL machine and run this command"
echo "   to create an SSH tunnel. This command will seem to hang, which is normal."
echo
echo "   ssh -N -L ${PORT}:$(hostname):${PORT} ${USER}@your_cluster_login_node"
echo
echo "   - Replace 'your_cluster_login_node' with your cluster's SSH address."
echo "   - The job is running on compute node: $(hostname)"
echo "----------------------------------------"
echo
echo "----------[ VS CODE ]----------"
echo "2. In VS Code, connect to the Jupyter server:"
echo "   a. Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P)."
echo "   b. Type and select 'Jupyter: Specify Jupyter server for connections'."
echo "   c. Select 'Existing'."
echo "   d. Paste one of the URLs below (it should start with http://127.0.0.1...)"
echo
echo "Available Jupyter Servers (copy a URL with the token):"
jupyter server list
echo
echo "3. Once connected, open your .ipynb file and select the kernel:"
echo "   a. Click the kernel name in the top-right corner of the notebook."
echo "   b. Choose 'SLURM Job Kernel' from the list."
echo "----------------------------------------"
echo
echo "Job is now running. The allocation is reserved for the time you requested."
echo "To stop the job, run: scancel $SLURM_JOB_ID"

# Wait for the Jupyter Lab process to end.
# This keeps the SLURM job alive.
wait
