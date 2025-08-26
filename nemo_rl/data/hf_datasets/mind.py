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


from typing import Any, Optional

from datasets import Dataset, load_dataset, load_from_disk

from nemo_rl.data.interfaces import TaskDataSpec


def format(
    data: dict[str, str | float | int], output_key: str = "messages"
) -> dict[str, list[Any] | str]:
    return {"messages": data[output_key]}


def prepare_mind_dataset(
    seed: int = 42,
    test_size: float = 0.01,
    dataset_name: str = ''
) -> dict[str, Dataset | None]:


    # Load the original dataset
    if 'swordhealth' in dataset_name:
        original_ds = load_dataset(dataset_name, split='train')
    else: 
        original_ds = load_from_disk(dataset_name)

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds.train_test_split(test_size=test_size, seed=seed)

    # Format the examples, removing original columns
    train_formatted = split_ds["train"].map(
        format,
        remove_columns=split_ds["train"].column_names,
    )
    val_formatted = split_ds["test"].map(
        format,
        remove_columns=split_ds["test"].column_names,
    )

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class MindDataset:
    def __init__(
        self,
        dataset_name,
        seed: int = 42,
        test_size: float = 0.01,
    ):
        """Initialize the Mind dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
        """
        

        self.formatted_ds = prepare_mind_dataset(dataset_name=dataset_name,
                                                 seed=seed, test_size=test_size,)

        self.task_spec = TaskDataSpec(
            task_name="Mind",
        )
