# {py:mod}`distributed.batched_data_dict`

```{py:module} distributed.batched_data_dict
```

```{autodoc2-docstring} distributed.batched_data_dict
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DynamicBatchingArgs <distributed.batched_data_dict.DynamicBatchingArgs>`
  - ```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs
    :summary:
    ```
* - {py:obj}`BatchedDataDict <distributed.batched_data_dict.BatchedDataDict>`
  -
* - {py:obj}`SlicedDataDict <distributed.batched_data_dict.SlicedDataDict>`
  - ```{autodoc2-docstring} distributed.batched_data_dict.SlicedDataDict
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DictT <distributed.batched_data_dict.DictT>`
  - ```{autodoc2-docstring} distributed.batched_data_dict.DictT
    :summary:
    ```
````

### API

````{py:data} DictT
:canonical: distributed.batched_data_dict.DictT
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} distributed.batched_data_dict.DictT
```

````

`````{py:class} DynamicBatchingArgs()
:canonical: distributed.batched_data_dict.DynamicBatchingArgs

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs
```

```{rubric} Initialization
```

```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs.__init__
```

````{py:attribute} max_tokens_per_microbatch
:canonical: distributed.batched_data_dict.DynamicBatchingArgs.max_tokens_per_microbatch
:type: int
:value: >
   None

```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs.max_tokens_per_microbatch
```

````

````{py:attribute} sequence_length_round
:canonical: distributed.batched_data_dict.DynamicBatchingArgs.sequence_length_round
:type: int
:value: >
   None

```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs.sequence_length_round
```

````

````{py:attribute} input_key
:canonical: distributed.batched_data_dict.DynamicBatchingArgs.input_key
:type: str
:value: >
   None

```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs.input_key
```

````

````{py:attribute} input_lengths_key
:canonical: distributed.batched_data_dict.DynamicBatchingArgs.input_lengths_key
:type: str
:value: >
   None

```{autodoc2-docstring} distributed.batched_data_dict.DynamicBatchingArgs.input_lengths_key
```

````

`````

`````{py:class} BatchedDataDict(*args, **kwargs)
:canonical: distributed.batched_data_dict.BatchedDataDict

Bases: {py:obj}`collections.UserDict`, {py:obj}`typing.Generic`\[{py:obj}`distributed.batched_data_dict.DictT`\]

````{py:method} from_batches(batches: list[dict[typing.Any, typing.Any]], pad_value_dict: typing.Optional[dict[str, int]] = None) -> typing_extensions.Self
:canonical: distributed.batched_data_dict.BatchedDataDict.from_batches
:classmethod:

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.from_batches
```

````

````{py:method} all_gather(group: torch.distributed.ProcessGroup) -> typing_extensions.Self
:canonical: distributed.batched_data_dict.BatchedDataDict.all_gather

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.all_gather
```

````

````{py:method} chunk(rank: int, chunks: int) -> distributed.batched_data_dict.SlicedDataDict
:canonical: distributed.batched_data_dict.BatchedDataDict.chunk

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.chunk
```

````

````{py:method} reorder_data(reorded_indices: typing.List[int])
:canonical: distributed.batched_data_dict.BatchedDataDict.reorder_data

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.reorder_data
```

````

````{py:method} shard_by_batch_size(shards: int, batch_size: typing.Optional[int] = None, allow_uneven_shards: bool = False, dynamic_batching_args: typing.Optional[distributed.batched_data_dict.DynamicBatchingArgs] = None) -> list[distributed.batched_data_dict.SlicedDataDict] | tuple[list[distributed.batched_data_dict.SlicedDataDict], list[int]]
:canonical: distributed.batched_data_dict.BatchedDataDict.shard_by_batch_size

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.shard_by_batch_size
```

````

````{py:method} get_batch(batch_idx, batch_size) -> distributed.batched_data_dict.SlicedDataDict
:canonical: distributed.batched_data_dict.BatchedDataDict.get_batch

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.get_batch
```

````

````{py:method} slice(start: int, end: int) -> distributed.batched_data_dict.SlicedDataDict
:canonical: distributed.batched_data_dict.BatchedDataDict.slice

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.slice
```

````

````{py:method} repeat_interleave(num_repeats: int) -> typing_extensions.Self
:canonical: distributed.batched_data_dict.BatchedDataDict.repeat_interleave

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.repeat_interleave
```

````

````{py:method} truncate_tensors(dim: int, truncated_len: int)
:canonical: distributed.batched_data_dict.BatchedDataDict.truncate_tensors

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.truncate_tensors
```

````

````{py:method} make_microbatch_iterator_with_dynamic_shapes(sequence_dim: int = 1) -> typing.Iterator[distributed.batched_data_dict.SlicedDataDict]
:canonical: distributed.batched_data_dict.BatchedDataDict.make_microbatch_iterator_with_dynamic_shapes

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.make_microbatch_iterator_with_dynamic_shapes
```

````

````{py:method} get_microbatch_iterator_dynamic_shapes_len() -> int
:canonical: distributed.batched_data_dict.BatchedDataDict.get_microbatch_iterator_dynamic_shapes_len

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.get_microbatch_iterator_dynamic_shapes_len
```

````

````{py:method} make_microbatch_iterator(microbatch_size: int) -> typing.Iterator[distributed.batched_data_dict.SlicedDataDict]
:canonical: distributed.batched_data_dict.BatchedDataDict.make_microbatch_iterator

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.make_microbatch_iterator
```

````

````{py:property} size
:canonical: distributed.batched_data_dict.BatchedDataDict.size
:type: int

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.size
```

````

````{py:method} to(device: str | torch.device) -> typing_extensions.Self
:canonical: distributed.batched_data_dict.BatchedDataDict.to

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.to
```

````

````{py:method} select_indices(indices: typing.Union[list[int], torch.Tensor]) -> typing_extensions.Self
:canonical: distributed.batched_data_dict.BatchedDataDict.select_indices

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.select_indices
```

````

````{py:method} get_dict() -> dict[typing.Any, typing.Any]
:canonical: distributed.batched_data_dict.BatchedDataDict.get_dict

```{autodoc2-docstring} distributed.batched_data_dict.BatchedDataDict.get_dict
```

````

`````

````{py:class} SlicedDataDict(*args, **kwargs)
:canonical: distributed.batched_data_dict.SlicedDataDict

Bases: {py:obj}`distributed.batched_data_dict.BatchedDataDict`

```{autodoc2-docstring} distributed.batched_data_dict.SlicedDataDict
```

```{rubric} Initialization
```

```{autodoc2-docstring} distributed.batched_data_dict.SlicedDataDict.__init__
```

````
