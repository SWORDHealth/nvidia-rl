

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data.datasets.utils import pil_to_base64
from nemo_rl.data.interfaces import TaskDataSpec


def format_medpix_vqa_dataset(
    example: dict[str, Any], return_pil: bool = False
) -> dict[str, Any]:
    """Format the MedPix-VQA dataset into an OpenAI-API-like message log."""
    
    user_content = [
        {
            "type": "image",
            "image": pil_to_base64(example["image_id"])
            if not return_pil
            else example["image_id"],
        },
        {
            "type": "text",
            "text": str(example["question"]),
        },
    ]

    assistant_content = str(example["answer"]).strip()

    ret = {
        "messages": [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "task_name": "medpix-vqa",
    }
    return ret


def prepare_medpix_vqa_dataset(
    split: str = "train",
    dataset_name: str = "../MedPix-VQA/data",
    task_name: Optional[str] = None,
):
    if task_name is None:
        task_name = "medpix-vqa"

    raw = load_dataset(dataset_name)

    if split == "train":
        tr_dataset = raw["train"]
        val_dataset = raw["validation"]
    else:
        if split not in raw:
            raise ValueError(f"Split '{split}' not found. Available: {list(raw.keys())}")

        tr_dataset = raw[split]
        val_dataset = raw[split]

    # format - disable features to avoid schema conflicts
    tr_dataset = tr_dataset.add_column("task_name", [task_name] * len(tr_dataset))
    val_dataset = val_dataset.add_column("task_name", [task_name] * len(val_dataset))

    return {
        "train": tr_dataset,
        "validation": val_dataset,
    }


class MedPixVQADataset:
    def __init__(
        self,
        dataset_name,
        split: str = "train",
    ):
        if split not in ["train", "validation"]:
            raise ValueError(
                f"Invalid split: {split}. Please use 'train' or 'validation."
            )
        self.task_name = "medpix-vqa"

        self.formatted_ds = prepare_medpix_vqa_dataset(
            split=split,
            task_name=self.task_name,
            dataset_name=dataset_name,
        )

        self.task_spec = TaskDataSpec(
            task_name="medpix-vqa",
        )
