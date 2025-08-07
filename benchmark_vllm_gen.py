import json

import ray

from vllm import LLM, SamplingParams


@ray.remote
class VLLMWorker:
    def __init__(self, llm_kwargs: dict, sampling_params: dict):
        self.sampling_params = SamplingParams(**sampling_params)

        self.llm = LLM(**llm_kwargs)

    def generate(self, prompt_token_ids: list[int]):
        outputs = self.llm.generate(prompt_token_ids, self.sampling_params)
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

    with open("prompt_token_ids.json") as f:
        all_prompt_token_ids = json.load(f)

    futures = []
    for prompt, worker in zip(all_prompt_token_ids, workers):
        futures.append(worker.generate.remote({"prompt_token_ids": prompt}))

    responses = ray.get(futures)

    for i, r in enumerate(responses):
        print(f"[GPU {i}] {r}")
