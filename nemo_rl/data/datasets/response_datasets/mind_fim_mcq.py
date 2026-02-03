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
import string

from nemo_rl.data.datasets.utils import load_dataset_from_path
from nemo_rl.data.interfaces import TaskDataSpec


def convert_options_to_letter_dict(
    options: list[dict[str, Any]],
) -> tuple[dict[str, str], str]:
    """Convert options list to a lettered dict and return the correct letter."""
    if not options:
        raise ValueError("Options list is empty.")
    if len(options) > len(string.ascii_uppercase):
        raise ValueError(
            f"Too many options ({len(options)}). Max supported is {len(string.ascii_uppercase)}."
        )

    option_dict: dict[str, str] = {}
    correct_letter: Optional[str] = None
    for idx, option in enumerate(options):
        letter = string.ascii_uppercase[idx]
        option_dict[letter] = str(option.get("text", ""))
        if option.get("is_correct") is True:
            if correct_letter is not None:
                raise ValueError("Multiple correct options found.")
            correct_letter = letter

    if correct_letter is None:
        raise ValueError("No correct option found.")

    return option_dict, correct_letter


class MindFIMMCQDataset:
    """Dataset class for Mind FIM multiple-choice questions."""

    def __init__(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        train_split: Optional[str] = None,
        val_split: Optional[str] = None,
    ) -> None:
        train_ds = load_dataset_from_path(train_data_path, train_split)
        val_ds = (
            load_dataset_from_path(val_data_path, val_split)
            if val_data_path
            else None
        )

        train_ds = train_ds.map(self._format_example, with_indices=True)
        if val_ds:
            val_ds = val_ds.map(self._format_example, with_indices=True)

        self.formatted_ds = {
            "train": train_ds,
            "validation": val_ds,
        }
        self.task_spec = TaskDataSpec(task_name="fim_mcq")

    def _format_example(
        self, example: dict[str, Any], idx: int
    ) -> dict[str, Any]:
        masked_reference_solution = example.get("masked_reference_solution")
        if masked_reference_solution is None:
            raise KeyError(
                "Expected 'masked_reference_solution' in dataset example."
            )

        options = example.get("options") or []
        if not isinstance(options, list):
            raise ValueError(
                f"Expected 'options' to be a list at index {idx}, got {type(options)}."
            )

        normalized_options = options
        if not any(option.get("is_correct") is True for option in options):
            removed_steps = example.get("removed_steps")
            if removed_steps is not None:
                removed_text = str(removed_steps).strip()
                normalized_options = []
                for option in options:
                    option_copy = dict(option)
                    option_text = str(option_copy.get("text", "")).strip()
                    if option_text == removed_text:
                        option_copy["is_correct"] = True
                    normalized_options.append(option_copy)

        try:
            option_dict, answer_letter = convert_options_to_letter_dict(
                normalized_options
            )
        except ValueError as exc:
            raise ValueError(f"Invalid options at index {idx}.") from exc

        question = f"Fill in the [MASK]: {masked_reference_solution}"
        return {
            "question": question,
            "options": option_dict,
            "answer": answer_letter,
            "task_name": "fim_mcq",
        }
