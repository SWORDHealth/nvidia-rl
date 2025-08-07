import json

from subprocess import Popen

from pydantic import BaseModel

from vllm import LLM


class VLLMInstanceMapping(BaseModel):
    local_rank: int  # Which GPU to start on
    tensor_parallel_size: int


def get_mapping(tensor_parallel_size: int):
    return [
        VLLMInstanceMapping(local_rank=local_rank, tensor_parallel_size=tensor_parallel_size)
        for local_rank in range(0, 8, tensor_parallel_size)
    ]


NODE_MAPPING_8x1 = get_mapping(tensor_parallel_size=1)
NODE_MAPPING_4x2 = get_mapping(tensor_parallel_size=2)
NODE_MAPPING_2x4 = get_mapping(tensor_parallel_size=4)
NODE_MAPPING_1x8 = get_mapping(tensor_parallel_size=8)


if __name__ == "__main__":
    with open("vllm_args.json") as f:
        vllm_args = json.load(f)
    Popen()
