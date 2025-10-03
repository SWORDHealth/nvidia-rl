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
from typing import Optional

from datasets import load_dataset, load_from_disk
from nemo_rl.data.interfaces import TaskDataSpec


class MindRMDataset:
    """Dataset class for preference data which can be loaded from a JSON file.

    This class handles loading of preference data for DPO and RM training.
    The input JSONL files should contain valid JSON objects formatted like this:
    {
        "context": list of dicts, # The prompt message (including previous turns, if any)
        "completions": list of dicts, # The list of completions
            {
                "rank": int, # The rank of the completion (lower rank is preferred)
                "completion": list of dicts, # The completion message(s)
            }
    }

    Args:
        dataset_name: Name of the HuggingFace dataset to load
        val_split_ratio: Fraction of data to use for validation (default: 0.1)
    """

    def __init__(
        self,
        dataset_name: str,
        val_split_ratio: float = 0.1,
    ):
        if 'swordhealth' in dataset_name:
            dataset = load_dataset(dataset_name)['train']
        else:
            dataset = load_from_disk(dataset_name)

        # Create train/val split
        split_dataset = dataset.train_test_split(test_size=val_split_ratio, seed=42)
        train_ds = split_dataset['train']
        val_ds = split_dataset['test']

        # store the formatted dataset
        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }

        self.task_spec = TaskDataSpec(task_name="PreferenceDataset")