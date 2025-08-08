from nemo_skills.pipeline.cli import wrap_arguments, sft_nemo_rl



# ------------------------------------------------------------
# HYPERPARAMETERS
# ------------------------------------------------------------
model_name = "LLaDA-8B-Instruct"
dataset_name = "gsm8k"

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------
data_dir = "/workspace/data"
log_dir = "/workspace/logs"

# ------------------------------------------------------------
# Project and Resources
# ------------------------------------------------------------
cluster = "nrt"

project_name = 'mdlm-sft'
exp_name = f'{model_name}-{dataset_name}'

nodes = 1
num_gpus = 8
chain_len = 1 # number of experiments to launch sequentially
# ------------------------------------------------------------



# RUN DAPO
def run_sft(chain_idx):
    _exp_name_base = f"{exp_name}"
    _exp_name = f"{_exp_name_base}-chain{chain_idx}"
    _run_after = f"{_exp_name_base}-chain{chain_idx-1}" if chain_idx > 0 else None
    sft_nemo_rl(
        ctx=wrap_arguments(
            f"++policy.is_mdl=true "
            f"++data.dataset_name=prompt_response_dataset "
            f"++data.input_key=question "
            f"++data.output_key=answer"
        ),
        cluster=cluster,
        run_after=_run_after,
        wandb_project=project_name,
        expname=_exp_name,
        output_dir=f"{log_dir}/{_exp_name_base}",
        hf_model=f"/hf_models/{model_name}",
        training_data=f"{data_dir}/{dataset_name}/train.jsonl",
        validation_data=f"{data_dir}/{dataset_name}/test.jsonl",
        num_gpus=num_gpus,
        num_nodes=nodes,
        num_training_jobs=1,
    )

for chain_idx in range(chain_len):
    run_sft(chain_idx)