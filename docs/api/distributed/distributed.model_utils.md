# {py:mod}`distributed.model_utils`

```{py:module} distributed.model_utils
```

```{autodoc2-docstring} distributed.model_utils
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DistributedLogprob <distributed.model_utils.DistributedLogprob>`
  - ```{autodoc2-docstring} distributed.model_utils.DistributedLogprob
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_distributed_log_softmax <distributed.model_utils._compute_distributed_log_softmax>`
  - ```{autodoc2-docstring} distributed.model_utils._compute_distributed_log_softmax
    :summary:
    ```
* - {py:obj}`from_parallel_logits_to_logprobs <distributed.model_utils.from_parallel_logits_to_logprobs>`
  - ```{autodoc2-docstring} distributed.model_utils.from_parallel_logits_to_logprobs
    :summary:
    ```
````

### API

````{py:function} _compute_distributed_log_softmax(vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor
:canonical: distributed.model_utils._compute_distributed_log_softmax

```{autodoc2-docstring} distributed.model_utils._compute_distributed_log_softmax
```
````

`````{py:class} DistributedLogprob
:canonical: distributed.model_utils.DistributedLogprob

Bases: {py:obj}`torch.autograd.Function`

```{autodoc2-docstring} distributed.model_utils.DistributedLogprob
```

````{py:method} forward(ctx: typing.Any, vocab_parallel_logits: torch.Tensor, target: torch.Tensor, vocab_start_index: int, vocab_end_index: int, group: torch.distributed.ProcessGroup, inference_only: bool = False) -> torch.Tensor
:canonical: distributed.model_utils.DistributedLogprob.forward
:staticmethod:

```{autodoc2-docstring} distributed.model_utils.DistributedLogprob.forward
```

````

````{py:method} backward(ctx: typing.Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None, None, None, None, None, None]
:canonical: distributed.model_utils.DistributedLogprob.backward
:staticmethod:

```{autodoc2-docstring} distributed.model_utils.DistributedLogprob.backward
```

````

`````

````{py:function} from_parallel_logits_to_logprobs(vocab_parallel_logits: torch.Tensor, target: torch.Tensor, vocab_start_index: int, vocab_end_index: int, group: torch.distributed.ProcessGroup, inference_only: bool = False) -> torch.Tensor
:canonical: distributed.model_utils.from_parallel_logits_to_logprobs

```{autodoc2-docstring} distributed.model_utils.from_parallel_logits_to_logprobs
```
````
