# {py:mod}`algorithms.interfaces`

```{py:module} algorithms.interfaces
```

```{autodoc2-docstring} algorithms.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LossType <algorithms.interfaces.LossType>`
  -
* - {py:obj}`LossFunction <algorithms.interfaces.LossFunction>`
  - ```{autodoc2-docstring} algorithms.interfaces.LossFunction
    :summary:
    ```
````

### API

`````{py:class} LossType
:canonical: algorithms.interfaces.LossType

Bases: {py:obj}`enum.Enum`

````{py:attribute} TOKEN_LEVEL
:canonical: algorithms.interfaces.LossType.TOKEN_LEVEL
:value: >
   'token_level'

```{autodoc2-docstring} algorithms.interfaces.LossType.TOKEN_LEVEL
```

````

````{py:attribute} SEQUENCE_LEVEL
:canonical: algorithms.interfaces.LossType.SEQUENCE_LEVEL
:value: >
   'sequence_level'

```{autodoc2-docstring} algorithms.interfaces.LossType.SEQUENCE_LEVEL
```

````

`````

`````{py:class} LossFunction
:canonical: algorithms.interfaces.LossFunction

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} algorithms.interfaces.LossFunction
```

````{py:attribute} loss_type
:canonical: algorithms.interfaces.LossFunction.loss_type
:type: algorithms.interfaces.LossType
:value: >
   None

```{autodoc2-docstring} algorithms.interfaces.LossFunction.loss_type
```

````

````{py:method} __call__(next_token_logits: torch.Tensor, data: nemo_rl.distributed.batched_data_dict.BatchedDataDict, global_valid_seqs: torch.Tensor, global_valid_toks: torch.Tensor) -> tuple[torch.Tensor, dict[str, typing.Any]]
:canonical: algorithms.interfaces.LossFunction.__call__

```{autodoc2-docstring} algorithms.interfaces.LossFunction.__call__
```

````

`````
