import json

import os

import ray

from vllm import LLM, SamplingParams


@ray.remote
class VLLMWorker:
    def __init__(self, llm_kwargs: dict, env_vars: dict, sampling_params: dict):
        self.sampling_params = SamplingParams(**self.sampling_params)

        print("Initial CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
        for k, v in env_vars.items():
            os.environ[k] = v
        print("Final CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.llm = LLM(**llm_kwargs)

    def generate(self, prompt: str):
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text


if __name__ == "__main__":
    with open("vllm_args.json") as f:
        vllm_args = json.load(f)
    llm_kwargs = vllm_args["vllm_llm_kwargs"]
    env_vars = vllm_args["environment_variables"]

    NUM_GPUS = 8
    TENSOR_PARALLEL_SIZE = 1

    workers = []
    for _ in range(NUM_GPUS // TENSOR_PARALLEL_SIZE):
        worker = VLLMWorker.options(num_gpus=TENSOR_PARALLEL_SIZE).remote(llm_kwargs, env_vars, dict())
        workers.append(worker)

    prompt = "Explain Ray in one sentence."
    futures = [worker.generate.remote(prompt) for worker in workers]
    responses = ray.get(futures)

    for i, r in enumerate(responses):
        print(f"[GPU {i}] {r}")
