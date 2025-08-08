# {py:mod}`utils.checkpoint`

```{py:module} utils.checkpoint
```

```{autodoc2-docstring} utils.checkpoint
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CheckpointingConfig <utils.checkpoint.CheckpointingConfig>`
  - ```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig
    :summary:
    ```
* - {py:obj}`CheckpointManager <utils.checkpoint.CheckpointManager>`
  - ```{autodoc2-docstring} utils.checkpoint.CheckpointManager
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_load_checkpoint_history <utils.checkpoint._load_checkpoint_history>`
  - ```{autodoc2-docstring} utils.checkpoint._load_checkpoint_history
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PathLike <utils.checkpoint.PathLike>`
  - ```{autodoc2-docstring} utils.checkpoint.PathLike
    :summary:
    ```
````

### API

````{py:data} PathLike
:canonical: utils.checkpoint.PathLike
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.PathLike
```

````

`````{py:class} CheckpointingConfig()
:canonical: utils.checkpoint.CheckpointingConfig

Bases: {py:obj}`typing.TypedDict`

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.__init__
```

````{py:attribute} enabled
:canonical: utils.checkpoint.CheckpointingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.enabled
```

````

````{py:attribute} checkpoint_dir
:canonical: utils.checkpoint.CheckpointingConfig.checkpoint_dir
:type: utils.checkpoint.PathLike
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.checkpoint_dir
```

````

````{py:attribute} metric_name
:canonical: utils.checkpoint.CheckpointingConfig.metric_name
:type: str
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.metric_name
```

````

````{py:attribute} higher_is_better
:canonical: utils.checkpoint.CheckpointingConfig.higher_is_better
:type: bool
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.higher_is_better
```

````

````{py:attribute} save_period
:canonical: utils.checkpoint.CheckpointingConfig.save_period
:type: int
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.save_period
```

````

````{py:attribute} keep_top_k
:canonical: utils.checkpoint.CheckpointingConfig.keep_top_k
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} utils.checkpoint.CheckpointingConfig.keep_top_k
```

````

`````

`````{py:class} CheckpointManager(config: utils.checkpoint.CheckpointingConfig)
:canonical: utils.checkpoint.CheckpointManager

```{autodoc2-docstring} utils.checkpoint.CheckpointManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.__init__
```

````{py:method} init_tmp_checkpoint(step: int, training_info: dict[str, typing.Any], run_config: typing.Optional[dict[str, typing.Any]] = None) -> utils.checkpoint.PathLike
:canonical: utils.checkpoint.CheckpointManager.init_tmp_checkpoint

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.init_tmp_checkpoint
```

````

````{py:method} finalize_checkpoint(checkpoint_path: utils.checkpoint.PathLike) -> None
:canonical: utils.checkpoint.CheckpointManager.finalize_checkpoint

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.finalize_checkpoint
```

````

````{py:method} remove_old_checkpoints(exclude_latest: bool = True) -> None
:canonical: utils.checkpoint.CheckpointManager.remove_old_checkpoints

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.remove_old_checkpoints
```

````

````{py:method} get_best_checkpoint_path() -> typing.Optional[str]
:canonical: utils.checkpoint.CheckpointManager.get_best_checkpoint_path

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.get_best_checkpoint_path
```

````

````{py:method} get_latest_checkpoint_path() -> typing.Optional[str]
:canonical: utils.checkpoint.CheckpointManager.get_latest_checkpoint_path

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.get_latest_checkpoint_path
```

````

````{py:method} load_training_info(checkpoint_path: typing.Optional[utils.checkpoint.PathLike] = None) -> typing.Optional[dict[str, typing.Any]]
:canonical: utils.checkpoint.CheckpointManager.load_training_info

```{autodoc2-docstring} utils.checkpoint.CheckpointManager.load_training_info
```

````

`````

````{py:function} _load_checkpoint_history(checkpoint_dir: pathlib.Path) -> list[tuple[int, utils.checkpoint.PathLike, dict[str, typing.Any]]]
:canonical: utils.checkpoint._load_checkpoint_history

```{autodoc2-docstring} utils.checkpoint._load_checkpoint_history
```
````
