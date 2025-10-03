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

from absl import logging
from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec

def to_preference_data_format(
    data: dict[str, Any],
) -> dict[
    str, list[dict[str, int | list[dict[str, str | Any]]]] | list[dict[str, str]]
]:
    chosen = data["chosen"]
    rejected = data["rejected"]

    return {
        "context": [{"role": "user", "content": data["prompt"]}]
        if isinstance(data["prompt"], str)
        else data["prompt"],
        "completions": [
            {"rank": 0, "completion": [{"role": "assistant", "content": chosen}]},
            {"rank": 1, "completion": [{"role": "assistant", "content": rejected}]},
        ],
    }


class MindDPODataset:

    def __init__(self, dataset_name: str, val_split_ratio: float = 0.01) -> None:
        ds = load_dataset(dataset_name)['train']

        # Create train/val split
        split_dataset = ds.train_test_split(test_size=val_split_ratio, seed=42)
        train_ds = split_dataset['train']
        val_ds = split_dataset['test']

        # Apply data format transformation
        train_ds = train_ds.map(to_preference_data_format, remove_columns=train_ds.column_names)
        val_ds = val_ds.map(to_preference_data_format, remove_columns=val_ds.column_names)

        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(
            task_name="MindDPO",
            )
