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

"""Contains data processors for evaluation."""

from typing import Any, cast
import warnings

import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.data.llm_message_utils import get_formatted_message_log


TokenizerType = PreTrainedTokenizerBase

def grpo_mind_preprocessor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:

    # Support both "prompt" (older datasets) and "context" (Dawn-Gym format)
    prompt = datum_dict.get("prompt") or datum_dict.get("context") or datum_dict.get("messages")
    if prompt is None:
        raise KeyError(f"Expected 'prompt', 'context', or 'messages' key in datum_dict. Available keys: {list(datum_dict.keys())}")

    # Get tools if present (for Qwen tool calling format)
    # Support both "tools" (list) and "tools_json" (JSON string to preserve structure)
    tools = datum_dict.get("tools")
    if tools is None and "tools_json" in datum_dict:
        import json
        tools = json.loads(datum_dict["tools_json"])

    def check_and_add_think_token(text: str) -> str:
        """Check if the generation prompt contains a beginning think token, add if missing."""
        if "<think>" not in text:
            # Add think token if not present
            text += "\n<think>\n"

        return text

    # Handle case where prompt is a simple string (e.g., math problem)
    if isinstance(prompt, str):
        # Convert string to user message format
        prompt = prompt.replace('user: ', '')
        prompt = [{"role": "user", "content": prompt}]

    # Determine if we need generation prompt (last message is user)
    last_msg_is_user = (
        isinstance(prompt, list) and
        len(prompt) > 0 and
        isinstance(prompt[-1], dict) and
        prompt[-1].get("role") == "user"
    )

    # Process the entire conversation with get_formatted_message_log
    # This handles tools properly and adds generation prompt if needed
    # When adding generation prompt, don't add EOS token (model will generate from there)
    message_log = get_formatted_message_log(
        prompt, tokenizer, task_data_spec,
        add_generation_prompt=last_msg_is_user,
        add_eos_token=not last_msg_is_user,  # No EOS when we want model to generate
        tools=tools
    )

    # Add think token to the last message if it's a user message with generation prompt
    if last_msg_is_user and len(message_log) > 0:
        last_msg = message_log[-1]
        content = str(last_msg.get("content", ""))
        if "<think>" not in content:
            # Add think token
            new_content = content + "\n<think>\n"
            last_msg["content"] = new_content
            last_msg["token_ids"] = tokenizer(
                new_content, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        warnings.warn(
            f"Sequence length {length} exceeds max_seq_length {max_seq_length}. Ignoring example."
        )
        # make smaller and mask out
        for message in message_log:
            message["token_ids"] = message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]

        loss_multiplier = 0.0

    # Prepare extra_env_info with ground_truth, dataset type, and rubric if present
    extra_env_info = {}
    if "ground_truth" in datum_dict:
        extra_env_info["ground_truth"] = datum_dict["ground_truth"]
    if "dataset" in datum_dict:
        extra_env_info["dataset"] = datum_dict["dataset"]
    # Support per-prompt custom rubrics for the LLM judge
    if "rubric" in datum_dict:
        extra_env_info["rubric"] = datum_dict["rubric"]
    # Support per-prompt max conversation turns
    if "max_turns" in datum_dict:
        extra_env_info["max_turns"] = datum_dict["max_turns"]
    # Support per-prompt user LLM system prompt (patient persona)
    if "user_system_prompt" in datum_dict:
        extra_env_info["user_system_prompt"] = datum_dict["user_system_prompt"]
    # Support per-prompt env_info (checklist_fns, non_negotiable_items, etc.)
    if "env_info" in datum_dict:
        extra_env_info["env_info"] = datum_dict["env_info"]
    if not extra_env_info:
        extra_env_info = None

    output = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }

    # Preserve task_name field if present in the input
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]

    return output


# Example of a generic math data processor
def math_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for the Math Environment."""
    problem = datum_dict["problem"]
    solution = str(datum_dict["expected_answer"])
    extra_env_info = {"ground_truth": solution}

    message_log: LLMMessageLogType = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        problem = task_data_spec.prompt.format(problem)
    user_message = {"role": "user", "content": problem}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for indiv_message in message_log:
            indiv_message["token_ids"] = indiv_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output


def math_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from data/hf_datasets/openmathinstruct2.py) into a DatumSpec for the Reward Model Environment."""
    user_message = datum_dict["messages"]
    problem = user_message[0]["content"]
    extra_env_info = {"ground_truth": user_message[1]["content"]}

    message_log: LLMMessageLogType = []
    formatted_content = (
        task_data_spec.prompt.format(problem) if task_data_spec.prompt else problem
    )
    user_message = {
        "role": "user",
        "content": formatted_content,
    }
    message: list[str] = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(
        message,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)

    loss_multiplier = 1.0
    if length > max_seq_length:
        # make smaller and mask out
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }
    return output


def _construct_multichoice_prompt(
    prompt: str, question: str, options: dict[str, str]
) -> str:
    """Construct prompt from question and options."""
    output = prompt
    output += f"\n\nQuestion: {question}\nOptions:\n"
    output += "\n".join(
        [
            f"{letter}) {option}"
            for letter, option in options.items()
            if option is not None
        ]
    )
    return output


def multichoice_qa_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process a datum dictionary (directly loaded from dataset) into a DatumSpec for multiple-choice problems."""
    question = datum_dict["question"]
    answer = str(datum_dict["answer"])
    options = datum_dict["options"]
    extra_env_info = {"ground_truth": answer}
    if "subject" in datum_dict:
        extra_env_info.update({"subject": datum_dict["subject"]})

    message_log = []

    # system prompt
    if task_data_spec.system_prompt:
        sys_prompt: dict[str, str | torch.Tensor] = {
            "role": "system",
            "content": task_data_spec.system_prompt,
        }
        sys = tokenizer.apply_chat_template(
            [cast(dict[str, str], sys_prompt)],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        sys_prompt["token_ids"] = tokenizer(
            sys, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
        message_log.append(sys_prompt)

    # user prompt
    if task_data_spec.prompt:
        question = _construct_multichoice_prompt(
            task_data_spec.prompt, question, options
        )
    user_message = {"role": "user", "content": question}
    message = tokenizer.apply_chat_template(
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    user_message["token_ids"] = tokenizer(
        message, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    user_message["content"] = message
    message_log.append(user_message)

    length = sum(len(m["token_ids"]) for m in message_log)
    output: DatumSpec = {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": 1.0,
        "idx": idx,
    }
    if "task_name" in datum_dict:
        output["task_name"] = datum_dict["task_name"]
    return output
