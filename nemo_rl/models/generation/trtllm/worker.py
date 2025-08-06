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

from typing import Any, NotRequired, Optional, TypedDict, cast

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationOutputSpec,
)


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
        bundle_indices: Optional[list[int]] = None,  # not used now
        fraction_of_gpus: float = 1.0,  # not used now
        seed: Optional[int] = None,  # not used now
    ):
        """Initialize a TRTLLM worker for distributed inference.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
                          Only needed for the first worker in each tied worker group.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
        """
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.8)

        llm_kwargs = dict(
            model=config["model_name"],
            executor_type="ray",
            kv_cache_config=kv_cache_config,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=None,
        )

        self.llm = LLM(**llm_kwargs)

    def post_init(self):
        pass

    def report_device_id(self) -> list[str]:
        list_of_worker_results = self.llm.collective_rpc("report_device_id")
        return cast(list[str], list_of_worker_results)

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        pass

    def update_weights_from_ipc_handles(self, ipc_handles: dict[str, Any]) -> bool:
        # fail here, better to put it to an extension worker
        try:
            self.llm.collective_rpc("update_weights_from_ipc_handles", (ipc_handles, ))
            return True
        except Exception as e:
            print(f"Error during update weights: {e}")
            return False

    def sleep(self):
        self.llm.collective_rpc("reset_prefix_cache")
        # self.llm.collective_rpc("sleep")

    def wake_up(self, **kwargs):
        pass

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        pass
        # prompts = [
        #     "Hello, my name is",
        #     "The president of the United States is",
        #     "The capital of France is",
        #     "The future of AI is",
        # ]

        # llm_ret = self.llm.generate(prompts)
        # outputs = []
        # for index, r in enumerate(llm_ret):
        #     outputs.append(prompts[index] + " " + r.outputs[0].text)

        # print(f"{outputs=}")
