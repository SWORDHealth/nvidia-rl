# {py:mod}`distributed.collectives`

```{py:module} distributed.collectives
```

```{autodoc2-docstring} distributed.collectives
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`rebalance_nd_tensor <distributed.collectives.rebalance_nd_tensor>`
  - ```{autodoc2-docstring} distributed.collectives.rebalance_nd_tensor
    :summary:
    ```
* - {py:obj}`gather_jagged_object_lists <distributed.collectives.gather_jagged_object_lists>`
  - ```{autodoc2-docstring} distributed.collectives.gather_jagged_object_lists
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`T <distributed.collectives.T>`
  - ```{autodoc2-docstring} distributed.collectives.T
    :summary:
    ```
````

### API

````{py:data} T
:canonical: distributed.collectives.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} distributed.collectives.T
```

````

````{py:function} rebalance_nd_tensor(tensor: torch.Tensor, group: typing.Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor
:canonical: distributed.collectives.rebalance_nd_tensor

```{autodoc2-docstring} distributed.collectives.rebalance_nd_tensor
```
````

````{py:function} gather_jagged_object_lists(local_objects: list[distributed.collectives.T], group: typing.Optional[torch.distributed.ProcessGroup] = None) -> list[distributed.collectives.T]
:canonical: distributed.collectives.gather_jagged_object_lists

```{autodoc2-docstring} distributed.collectives.gather_jagged_object_lists
```
````
