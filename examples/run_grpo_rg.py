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
import itertools
import os
import pprint
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterator

import jsonlines
from omegaconf import OmegaConf
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.hf_datasets.deepscaler import DeepScalerDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.llm_judge_async_environment import LLMJudgeAsyncEnvironment
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.environments.ifeval_environment import IFEvalEnvironment
from nemo_rl.environments.reasoning_gym_env import (
    ReasoningGymEnv,
    ReasoningGymGameLogic,
    ReasoningGymMetadata,
)
from reasoning_gym.coaching.curriculum_config import CurriculumExperimentConfig
from reasoning_gym.coaching.experiment import CurriculumExperiment
from reasoning_gym.factory import has_curriculum
from nemo_rl.models.generation.interfaces import configure_generation_config
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.utils.logger import get_next_experiment_dir

OmegaConf.register_new_resolver("mul", lambda a, b: a * b)

def hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Math Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []
    user_message = {
        "role": "user",
        "content": task_data_spec.prompt.format(problem),
    }
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    args, overrides = parser.parse_known_args()

    return args, overrides


def filter_reasoning_gym_tasks(rg_config: dict) -> list[dict]:
    all_tasks = rg_config.get("all_tasks", {})
    selection = rg_config.get("task_selection", {"mode": "all"})
    mode = selection.get("mode", "all")
    
    flattened_tasks = []
    task_to_category = {}  
    
    for category, tasks in all_tasks.items():
        for task in tasks:
            flattened_tasks.append(task)
            task_to_category[task["dataset_name"]] = category
    
    if mode == "all":
        filtered_tasks = flattened_tasks
    
    elif mode == "categories":
        selected_categories = set(selection.get("categories", []))
        if not selected_categories:
            raise ValueError("No categories specified for mode='categories'")
        
        filtered_tasks = []
        for category, tasks in all_tasks.items():
            if category in selected_categories:
                filtered_tasks.extend(tasks)
        
        if not filtered_tasks:
            raise ValueError(f"No tasks found for categories: {selected_categories}")
    
    elif mode == "tasks":
        selected_tasks = set(selection.get("tasks", []))
        if not selected_tasks:
            raise ValueError("No tasks specified for mode='tasks'")
        
        filtered_tasks = [
            task for task in flattened_tasks 
            if task["dataset_name"] in selected_tasks
        ]
        
        if not filtered_tasks:
            raise ValueError(f"No tasks found with names: {selected_tasks}")
    
    elif mode == "exclude":
        excluded_categories = set(selection.get("exclude_categories", []))
        excluded_tasks = set(selection.get("exclude_tasks", []))
        
        filtered_tasks = []
        for task in flattened_tasks:
            task_name = task["dataset_name"]
            category = task_to_category[task_name]
            
            if category in excluded_categories or task_name in excluded_tasks:
                continue
            
            filtered_tasks.append(task)
        
        if not filtered_tasks:
            raise ValueError("All tasks were excluded!")
    
    else:
        raise ValueError(f"Unknown task selection mode: {mode}")
    
    # dynamic even weighting
    if filtered_tasks:
        even_weight = 1.0 / len(filtered_tasks)
        for task in filtered_tasks:
            task["weight"] = even_weight
        
        print(f"Applied even weighting: {even_weight:.4f} per task")
    
    return filtered_tasks


@dataclass
class JsonlinesDataset:
    jsonl_path: str
    seed: int
    tokenizer: AutoTokenizer
    max_seq_length: int
    filter_long_samples: bool = False

    def __post_init__(self):
        self.data = self._load_data()

        idx_to_ignore = set()
        if self.filter_long_samples:
            for i, item in enumerate(self):
                if item["length"] > self.max_seq_length:
                    idx_to_ignore.add(i)
            print(f"found {len(idx_to_ignore)} long samples to ignore on dataset init")

        self.data = [item for i, item in enumerate(self.data) if i not in idx_to_ignore]

    def _load_data(self):
        with jsonlines.open(self.jsonl_path, "r") as reader:
            data = [line for line in reader]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> DatumSpec:
        data = self.data[idx]
        # support single turn for now
        assert len(data["messages"]) == 1
        single_message = data["messages"][0]

        message_log = []

        # this will also contain system prompt
        user_message = {"role": "user"}

        for m in single_message:
            # it's actually taking only the last user message's metadata
            if m["role"] == "user":
                # need to be deepcopy to avoid overwriting the original metadata
                extra_env_info = deepcopy(m["metadata"])

        message = self.tokenizer.apply_chat_template(
            single_message,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        user_message["token_ids"] = self.tokenizer.apply_chat_template(
            single_message,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=False,
            return_tensors="pt",
        )[0]
        user_message["content"] = message
        message_log.append(user_message)

        length = sum(len(m["token_ids"]) for m in message_log)

        output = {
            "message_log": message_log,
            "length": length,
            "extra_env_info": extra_env_info,
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": data["task_name"],
            "dataset": data["dataset"],
        }

        return output


def generate_reasoning_gym_datum(
    tokenizer,
    game_config: dict[str, Any],
    task_name: str,
    idx: int,
    add_system_prompt: bool,
) -> DatumSpec:
    """Generate a single reasoning-gym puzzle datum (prompt and metadata)."""
    
    game_state = ReasoningGymGameLogic.generate(game_config)
    puzzle_question = ReasoningGymGameLogic.init(game_state)
    
    prompt_instructions = (
        f"{puzzle_question}\n\n"
        f"Think carefully about this problem and provide your final answer.\n"
        f"After reasoning, output your answer on a new line using this format:\n"
        f"<answer>your_final_answer</answer>\n"
    )
    
    initial_prompt_content = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_instructions}],
        tokenize=False,
        add_system_prompt=add_system_prompt,
        add_generation_prompt=True,
        add_special_tokens=False,
    ).strip()
    
    # for length calculation
    tokenized_prompt = tokenizer(
        initial_prompt_content, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    
    message_log: LLMMessageLogType = [
        {
            "role": "user",
            "content": initial_prompt_content,
            "token_ids": tokenized_prompt,
        }
    ]
    
    metadata = ReasoningGymMetadata(
        puzzle_entry=game_state,
        dataset_name=game_state["dataset_name"]
    )
    
    datum: DatumSpec = {
        "message_log": message_log,
        "length": len(tokenized_prompt),
        "extra_env_info": metadata,
        "loss_multiplier": 1.0,
        "idx": idx,
        "task_name": task_name,
        "stop_strings": ["</answer>"],
    }
    
    return datum


class IterableReasoningGymDataset(IterableDataset):
    """An IterableDataset that generates reasoning-gym puzzle data indefinitely."""

    def __init__(
        self, tokenizer, game_config, task_name, add_system_prompt, length
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.game_config = game_config
        self.task_name = task_name
        self.add_system_prompt = add_system_prompt
        self.length = length
        
        self.curriculum_config = game_config.get("curriculum", {})
        self.curriculum_enabled = self.curriculum_config.get("enabled", False)
        self.curriculum_experiment = None
        self.puzzle_count = 0
        self.current_level = 0
        
        if self.curriculum_enabled and "tasks" in game_config:
            self._initialize_curriculum()

    def _initialize_curriculum(self):
        curricula = {}
        for task in self.game_config["tasks"]:
            dataset_name = task["dataset_name"]
            
            if not has_curriculum(dataset_name):
                print(f"Warning: Dataset '{dataset_name}' does not support curriculum, using fixed config")
                continue
                
            start_level = self.curriculum_config.get("start_level", 0)
            curricula[dataset_name] = {
                "attribute_levels": {"*": start_level},
                "weight": task.get("weight", 1.0)
            }
        
        if curricula:
            curriculum_exp_config = CurriculumExperimentConfig(curricula=curricula)
            curriculum_exp_config.validate()
            
            self.curriculum_experiment = CurriculumExperiment(
                name=f"{self.task_name}_curriculum",
                config=curriculum_exp_config,
                size=1,
                seed=None
            )
            
            self.current_level = start_level
            print(f"Initialized curriculum with {len(curricula)} datasets at level {start_level}")
        else:
            print("No curriculum-enabled datasets found, disabling curriculum")
            self.curriculum_enabled = False

    def _maybe_increment_curriculum(self):
        if not self.curriculum_enabled or not self.curriculum_experiment:
            return
            
        increment_every = self.curriculum_config.get("increment_every_n_puzzles", 1000)
        max_level = self.curriculum_config.get("max_level", None)
        
        if self.puzzle_count > 0 and self.puzzle_count % increment_every == 0:
            any_incremented = False
            for dataset_name, curriculum in self.curriculum_experiment.curricula.items():
                if max_level is not None and curriculum.get_max_level() >= max_level:
                    continue
                    
                if curriculum.increment_global_level():
                    any_incremented = True
                    new_level = curriculum.get_max_level()
                    print(f"Incremented curriculum for {dataset_name} to level {new_level} (puzzle #{self.puzzle_count})")
            
            if any_incremented:
                self.current_level += 1
                for dataset_name, curriculum in self.curriculum_experiment.curricula.items():
                    config = curriculum.get_global_level()
                    self.curriculum_experiment.composite.update_dataset_config(dataset_name, config)

    def __iter__(self) -> Iterator[DatumSpec]:
        if "tasks" in self.game_config:
            num_tasks = len(self.game_config['tasks'])
            dataset_info = f"{num_tasks} tasks: {', '.join([task['dataset_name'] for task in self.game_config['tasks']])}"
        else:
            dataset_info = "unknown configuration"
            
        print(f"Starting IterableReasoningGymDataset for {dataset_info} (indefinite generation).")
        
        if self.curriculum_enabled and self.curriculum_experiment:
            print(f"Curriculum enabled: increment every {self.curriculum_config.get('increment_every_n_puzzles', 1000)} puzzles")
        
        for i in itertools.count():
            if self.curriculum_enabled and self.curriculum_experiment:
                self._maybe_increment_curriculum()
                
                datum = self._generate_curriculum_datum(i)
            else:
                datum = generate_reasoning_gym_datum(
                    tokenizer=self.tokenizer,
                    game_config=self.game_config,
                    task_name=self.task_name,
                    idx=i,
                    add_system_prompt=self.add_system_prompt,
                )
            
            self.puzzle_count += 1
            yield datum

    def _generate_curriculum_datum(self, idx: int) -> DatumSpec:
        puzzle_entry = self.curriculum_experiment.get_dataset_entry(0)
        
        self.curriculum_experiment = CurriculumExperiment(
            name=f"{self.task_name}_curriculum",
            config=self.curriculum_experiment.curriculum_config,
            size=1,
            seed=None
        )
        
        for dataset_name, curriculum in self.curriculum_experiment.curricula.items():
            curriculum.set_global_level(self.current_level)
        
        puzzle_question = puzzle_entry["question"]
        prompt_instructions = (
            f"{puzzle_question}\n\n"
            f"Think carefully about this problem and provide your final answer.\n"
            f"After reasoning, output your answer on a new line using this format:\n"
            f"<answer>your_final_answer</answer>\n"
        )
        
        initial_prompt_content = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_instructions}],
            tokenize=False,
            add_system_prompt=self.add_system_prompt,
            add_generation_prompt=True,
            add_special_tokens=False,
        ).strip()
        
        tokenized_prompt = self.tokenizer(
            initial_prompt_content, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        
        message_log: LLMMessageLogType = [
            {
                "role": "user",
                "content": initial_prompt_content,
                "token_ids": tokenized_prompt,
            }
        ]
        
        metadata = ReasoningGymMetadata(
            puzzle_entry={
                "puzzle_entry": puzzle_entry,
                "dataset_name": puzzle_entry["metadata"]["source_dataset"],
                "dataset": self.curriculum_experiment.composite
            },
            dataset_name=puzzle_entry["metadata"]["source_dataset"]
        )
        
        datum: DatumSpec = {
            "message_log": message_log,
            "length": len(tokenized_prompt),
            "extra_env_info": metadata,
            "loss_multiplier": 1.0,
            "idx": idx,
            "task_name": self.task_name,
            "stop_strings": ["</answer>"],
            "curriculum_level": self.current_level,
        }
        
        return datum

    def __len__(self):
        return self.length


def setup_data(tokenizer: AutoTokenizer, data_config: DataConfig, env_configs):
    print("\n▶ Setting up data...")

    # if reasoning-gym is enabled, use it instead of jsonlines
    if "reasoning_gym_task" in env_configs and env_configs["reasoning_gym_task"]["enable"]:
        print("Reasoning Gym enabled - using IterableReasoningGymDataset")
        
        rg_config = env_configs["reasoning_gym_task"]
        
        # task filtering
        if "all_tasks" in rg_config:
            filtered_tasks = filter_reasoning_gym_tasks(rg_config)
            print(f"Selected {len(filtered_tasks)} tasks after filtering")
            
            task_names = [t["dataset_name"] for t in filtered_tasks]
            print(f"Tasks: {', '.join(task_names)}")
            
            game_config = {"tasks": filtered_tasks}
        elif "tasks" in rg_config:
            # pre-filter approach
            game_config = {"tasks": rg_config["tasks"]}
        else:
            raise ValueError("reasoning_gym_task config must specify 'tasks' or 'all_tasks'")
        
        if "curriculum" in rg_config:
            game_config["curriculum"] = rg_config["curriculum"]
            if game_config["curriculum"].get("enabled", False):
                print(f"Curriculum learning enabled: mode={game_config['curriculum'].get('mode', 'fixed')}, "
                      f"increment_every={game_config['curriculum'].get('increment_every_n_puzzles', 1000)}")
        
        train_ds = IterableReasoningGymDataset(
            tokenizer=tokenizer,
            game_config=game_config,
            task_name="reasoning_gym_task",
            add_system_prompt=data_config.get("add_system_prompt", False),
            length=10000, # really infinite gen
        )
        
        val_ds = IterableReasoningGymDataset(
            tokenizer=tokenizer,
            game_config=game_config,
            task_name="reasoning_gym_task", 
            add_system_prompt=data_config.get("add_system_prompt", False),
            length=1000,
        )
    else:
        train_ds = JsonlinesDataset(
            data_config["train"]["jsonl_path"],
            data_config["train"]["seed"],
            tokenizer,
            max_seq_length=data_config["max_input_seq_length"],
            filter_long_samples=data_config["train"]["filter_long_samples"],
        )
        val_ds = JsonlinesDataset(
            data_config["val"]["jsonl_path"],
            data_config["val"]["seed"],
            tokenizer,
            max_seq_length=data_config["max_input_seq_length"],
            filter_long_samples=data_config["val"]["filter_long_samples"],
        )

    # Create DeepScaler validation dataset for math evaluation
    print("Loading DeepScaler for validation...")
    deepscaler_data = DeepScalerDataset()
    math_task_spec = TaskDataSpec(task_name="math", prompt_file="examples/prompts/math.txt", system_prompt_file=None)
    
    # Set up task data processors for validation
    val_task_data_processors = defaultdict(lambda: (math_task_spec, hf_data_processor))
    val_task_data_processors["math"] = (math_task_spec, hf_data_processor)
    
    val_ds = AllTaskProcessedDataset(
        deepscaler_data.formatted_ds["validation"].select(range(100)),
        tokenizer,
        math_task_spec,
        val_task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )

    task_to_env = {}

    if "math" in env_configs and env_configs["math"]["enable"]:
        math_env = MathEnvironment.options(
            runtime_env={
                "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(
                    os.environ
                ),  # Pass thru all user environment variables
            }
        ).remote(env_configs["math"])
        task_to_env["math"] = math_env
    
    if "ifeval" in env_configs and env_configs["ifeval"]["enable"]:
        ifeval_env = IFEvalEnvironment.options(
            runtime_env={
                "py_executable": IFEvalEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs["ifeval"])
        task_to_env["ifeval"] = ifeval_env
        
    if "reasoning_gym_task" in env_configs and env_configs["reasoning_gym_task"]["enable"]:
        rg_config = env_configs["reasoning_gym_task"]
        if "all_tasks" in rg_config:
            filtered_tasks = filter_reasoning_gym_tasks(rg_config)
            game_config = {"tasks": filtered_tasks}
        elif "tasks" in rg_config:
            game_config = {"tasks": rg_config["tasks"]}
        else:
            raise ValueError("reasoning_gym_task config must specify 'tasks' or 'all_tasks'")
        
        if "curriculum" in rg_config:
            game_config["curriculum"] = rg_config["curriculum"]
            
        reasoning_gym_env = ReasoningGymEnv.options(
            num_gpus=0,
            runtime_env={
                "py_executable": ReasoningGymEnv.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            }
        ).remote(cfg=game_config)
        task_to_env["reasoning_gym_task"] = reasoning_gym_env
        
    if "llm_judge_async" in env_configs and env_configs["llm_judge_async"]["enable"]:
        # Extract max_concurrency from config, default to 16 if not specified
        max_concurrency = env_configs["llm_judge_async"].get("max_concurrency", 16)

        llm_judge_async_env = LLMJudgeAsyncEnvironment.options(
            max_concurrency=max_concurrency,
            runtime_env={
                "py_executable": LLMJudgeAsyncEnvironment.DEFAULT_PY_EXECUTABLE,
                "env_vars": dict(os.environ),
            },
        ).remote(env_configs["llm_judge_async"])
        task_to_env["llm_judge"] = llm_judge_async_env

    # Create math environment for validation
    math_env = MathEnvironment.options(
        runtime_env={
            "py_executable": MathEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote({"num_workers": 4})
    
    val_task_to_env = task_to_env.copy()
    val_task_to_env["math"] = math_env

    return train_ds, val_ds, task_to_env, val_task_to_env


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(os.path.dirname(__file__), "configs", "grpo_1B.yaml")

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
    print(f"📊 Using log directory: {config['logger']['log_dir']}")
    if config["checkpointing"]["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config['checkpointing']['checkpoint_dir']}"
        )

    init_ray()

    # setup tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # setup data
    (
        dataset,
        val_dataset,
        task_to_env,
        val_task_to_env,
    ) = setup_data(tokenizer, config["data"], config["env"])

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


if __name__ == "__main__":
    main()
