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


from typing import Any

from datasets import Dataset, load_dataset, load_from_disk

from nemo_rl.data.interfaces import TaskDataSpec
import os


def to_response_data_format(
    data: dict[str, Any],
) -> dict[
    str, list[dict[str, int | list[dict[str, str | Any]]]] | list[dict[str, str]]
]:
    # Handle different data formats - check if this is DPO-style data
    if "prompt" in data:
        prompt = data["prompt"]
    elif "context" in data and isinstance(data["context"], list):
        # Extract prompt from context messages
        prompt = data["context"]
    else:
        # Fallback - use the first available text field
        prompt = data.get("text", data.get("question", ""))

    result = {
        "context": [{"role": "user", "content": prompt}]
        if isinstance(prompt, str)
        else prompt,
        "task_name": "mind"
    }

    # Preserve ground_truth if it exists
    if "ground_truth" in data:
        result["ground_truth"] = data["ground_truth"]

    # Preserve dataset type if it exists (for routing math vs ifeval)
    if "dataset" in data:
        result["dataset"] = data["dataset"]

    return result


class MindGRPODataset:

    def __init__(self, dataset_name: str, val_split_ratio: float = 0.01) -> None:
        if 'swordhealth/' in dataset_name:
            ds = load_dataset(dataset_name)
        else:
            ds = load_from_disk(dataset_name)

        # Apply formatting to the dataset
        formatted_ds = ds.map(to_response_data_format)

        # Split into train and validation sets
        if isinstance(formatted_ds, dict):
            # If dataset has multiple splits, use 'train' split
            train_ds = formatted_ds['train'] if 'train' in formatted_ds else formatted_ds[list(formatted_ds.keys())[0]]
        else:
            train_ds = formatted_ds

        # Use entire dataset for training (no split)
        self.formatted_ds = {
            'train': train_ds,
            'validation': train_ds  # Use same data for validation
        }

        self.task_spec = TaskDataSpec(
            task_name="ResponseDataset",
        )