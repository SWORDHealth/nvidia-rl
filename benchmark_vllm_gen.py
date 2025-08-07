import json

import os

import ray

from vllm import LLM, SamplingParams


@ray.remote
class VLLMWorker:
    def __init__(self, llm_kwargs: dict, sampling_params: dict):
        self.sampling_params = SamplingParams(**sampling_params)

        self.llm = LLM(**llm_kwargs)

    def generate(self, prompt: str):
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text


if __name__ == "__main__":
    with open("vllm_args.json") as f:
        vllm_args = json.load(f)
    llm_kwargs = vllm_args["vllm_llm_kwargs"]
    sampling_params = vllm_args["sampling_params"]

    NUM_GPUS = 8
    TENSOR_PARALLEL_SIZE = 1

    workers = []
    for _ in range(NUM_GPUS // TENSOR_PARALLEL_SIZE):
        worker = VLLMWorker.options(num_gpus=TENSOR_PARALLEL_SIZE).remote(llm_kwargs, sampling_params)
        workers.append(worker)

    prompt = "Explain Ray in one sentence."
    futures = [worker.generate.remote(prompt) for worker in workers]
    responses = ray.get(futures)

    for i, r in enumerate(responses):
        print(f"[GPU {i}] {r}")

    with open("prompt_token_ids.json") as f:
        all_prompt_token_ids = json.load(f)
