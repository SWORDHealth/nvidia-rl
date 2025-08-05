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

import os
from typing import Any, NotRequired, Optional, TypedDict

import ray

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import GenerationConfig


class TRTLLMSpecificArgs(TypedDict):
    pass


class TRTLLMConfig(GenerationConfig):
    trtllm_cfg: TRTLLMSpecificArgs
    trtllm_kwargs: NotRequired[dict[str, Any]]


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("trtllm_generation_worker")}
)  # pragma: no cover
class TRTLLMGenerationWorker:
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        return f"{self.__class__.__name__}"

    def __init__(
        self,
        config: TRTLLMConfig,
        bundle_indices: Optional[list[int]] = None,
        fraction_of_gpus: float = 1.0,
        seed: Optional[int] = None,
    ):
        print(f"{os.environ["PATH"]=}")
        print(f"{os.environ["CUDA_VISIBLE_DEVICES"]=}")

        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)

        config = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "executor_type": "ray",
            "kv_cache_config": kv_cache_config,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": None,
        }

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        llm = LLM(**config)
        llm_ret = llm.generate(prompts)
        outputs = []
        for index, r in enumerate(llm_ret):
            outputs.append(prompts[index] + " " + r.outputs[0].text)

        print(f"{outputs=}")
        print("TRTLLM init success")

    def post_init(self):
        pass
