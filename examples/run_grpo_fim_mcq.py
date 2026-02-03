# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pprint
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional

from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.response_datasets.mind_fim_mcq import MindFIMMCQDataset
from nemo_rl.data.interfaces import (
    TaskDataProcessFnCallable,
    TaskDataSpec,
)
from nemo_rl.data.processors import fim_mcq_data_processor
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m {secs}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{seconds:.1f}s"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, overrides = parser.parse_known_args()

    return args, overrides


# ===============================================================================
#                             FIM MCQ Data Processor
# ===============================================================================
TokenizerType = PreTrainedTokenizerBase


def setup_data(
    tokenizer: TokenizerType,
    data_config: DataConfig,
    env_configs: dict[str, Any],
    seed: int,
) -> tuple[
    AllTaskProcessedDataset,
    Optional[AllTaskProcessedDataset],
    dict[str, EnvironmentInterface],
    dict[str, EnvironmentInterface],
]:
    print("\nSetting up data...")
    task_name = "fim_mcq"
    fim_mcq_task_spec = TaskDataSpec(
        task_name=task_name,
        prompt_file=data_config["prompt_file"],
        system_prompt_file=data_config["system_prompt_file"],
    )

    if "train_data_path" not in data_config:
        raise ValueError("train_data_path is required for MindFIMMCQDataset.")

    data = MindFIMMCQDataset(
        train_data_path=data_config["train_data_path"],
        val_data_path=data_config.get("val_data_path"),
        train_split=data_config.get("train_split"),
        val_split=data_config.get("val_split"),
    )

    task_data_processors: dict[str, tuple[TaskDataSpec, TaskDataProcessFnCallable]] = (
        defaultdict(lambda: (fim_mcq_task_spec, fim_mcq_data_processor))
    )
    task_data_processors[task_name] = (fim_mcq_task_spec, fim_mcq_data_processor)

    fim_mcq_env = MathEnvironment.options(  # type: ignore # it's wrapped with ray.remote
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            ),
            "env_vars": dict(os.environ),  # Pass thru all user environment variables
        }
    ).remote(env_configs[task_name])

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        fim_mcq_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    val_dataset: Optional[AllTaskProcessedDataset] = None
    if data.formatted_ds["validation"]:
        val_dataset = AllTaskProcessedDataset(
            data.formatted_ds["validation"],
            tokenizer,
            fim_mcq_task_spec,
            task_data_processors,
            max_seq_length=data_config["max_input_seq_length"],
        )
    else:
        val_dataset = None

    task_to_env: dict[str, EnvironmentInterface] = defaultdict(lambda: fim_mcq_env)
    task_to_env[task_name] = fim_mcq_env
    return dataset, val_dataset, task_to_env, task_to_env


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_mind_fim_mcq.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Get the next experiment directory with incremented ID
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    print(f"Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    assert config["policy"]["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"], config["grpo"]["seed"])

    (
        policy,
        policy_generation,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
    ) = setup(config, tokenizer, dataset, val_dataset)

    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    print("=" * 80)
    print(f"Training started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Check if async mode is enabled
    if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
        # Async GRPO does not support dynamic sampling, reward scaling, or reward shaping (DAPO features)
        unsupported_features = [
            "use_dynamic_sampling",
            "reward_scaling",
            "reward_shaping",
        ]

        for feature in unsupported_features:
            if feature not in config["grpo"]:
                continue

            if feature == "use_dynamic_sampling":
                if config["grpo"][feature]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )
            else:
                if config["grpo"][feature]["enabled"]:
                    raise NotImplementedError(
                        f"{feature} is not supported with async GRPO"
                    )

        from nemo_rl.algorithms.grpo import async_grpo_train

        print("Running async GRPO training")

        async_config = config["grpo"]["async_grpo"]
        # Run async GRPO training
        async_grpo_train(
            policy=policy,
            policy_generation=policy_generation,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            tokenizer=tokenizer,
            loss_fn=loss_fn,
            task_to_env=task_to_env,
            val_task_to_env=val_task_to_env,
            logger=logger,
            checkpointer=checkpointer,
            grpo_save_state=grpo_state,
            master_config=master_config,
            max_trajectory_age_steps=async_config["max_trajectory_age_steps"],
        )
    else:
        print("Running synchronous GRPO training")

        # Run standard GRPO training
        grpo_train(
            policy,
            policy_generation,
            dataloader,
            val_dataloader,
            tokenizer,
            loss_fn,
            task_to_env,
            val_task_to_env,
            logger,
            checkpointer,
            grpo_state,
            master_config,
        )

    # Record end time and calculate duration
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time

    print("=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:   {format_duration(total_duration)}")
    print(f"Total time: {total_duration:.1f} seconds")
    print("=" * 80)

    # Save timing information to log directory
    log_dir = config["logger"]["log_dir"]
    timing_file = os.path.join(log_dir, "training_timing.txt")

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Write timing info to file
    with open(timing_file, "w") as f:
        f.write("GRPO Training Timing Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time:   {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration:   {format_duration(total_duration)}\n")
        f.write(f"Total time: {total_duration:.1f} seconds\n")
        f.write(f"Max steps:  {config['grpo'].get('max_num_steps', 'Not set')}\n")
        f.write(f"Model:      {config['policy']['model_name']}\n")

        # Add async/sync mode info
        if "async_grpo" in config["grpo"] and config["grpo"]["async_grpo"]["enabled"]:
            f.write("Mode:       Async GRPO\n")
            f.write(
                f"Max trajectory age: {config['grpo']['async_grpo']['max_trajectory_age_steps']}\n"
            )
            f.write(
                "In-flight updates:  "
                f"{config['grpo']['async_grpo'].get('in_flight_weight_updates', False)}\n"
            )
        else:
            f.write("Mode:       Synchronous GRPO\n")

    print(f"Timing saved to: {timing_file}")


if __name__ == "__main__":
    main()
