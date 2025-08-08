# {py:mod}`utils.timer`

```{py:module} utils.timer
```

```{autodoc2-docstring} utils.timer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Timer <utils.timer.Timer>`
  - ```{autodoc2-docstring} utils.timer.Timer
    :summary:
    ```
````

### API

`````{py:class} Timer()
:canonical: utils.timer.Timer

```{autodoc2-docstring} utils.timer.Timer
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.timer.Timer.__init__
```

````{py:attribute} _REDUCTION_FUNCTIONS
:canonical: utils.timer.Timer._REDUCTION_FUNCTIONS
:type: dict[str, typing.Callable[[typing.Sequence[float]], float]]
:value: >
   None

```{autodoc2-docstring} utils.timer.Timer._REDUCTION_FUNCTIONS
```

````

````{py:method} start(label: str) -> None
:canonical: utils.timer.Timer.start

```{autodoc2-docstring} utils.timer.Timer.start
```

````

````{py:method} stop(label: str) -> float
:canonical: utils.timer.Timer.stop

```{autodoc2-docstring} utils.timer.Timer.stop
```

````

````{py:method} time(label: str) -> typing.Generator[None, None, None]
:canonical: utils.timer.Timer.time

```{autodoc2-docstring} utils.timer.Timer.time
```

````

````{py:method} get_elapsed(label: str) -> list[float]
:canonical: utils.timer.Timer.get_elapsed

```{autodoc2-docstring} utils.timer.Timer.get_elapsed
```

````

````{py:method} get_latest_elapsed(label: str) -> float
:canonical: utils.timer.Timer.get_latest_elapsed

```{autodoc2-docstring} utils.timer.Timer.get_latest_elapsed
```

````

````{py:method} reduce(label: str, operation: str = 'mean') -> float
:canonical: utils.timer.Timer.reduce

```{autodoc2-docstring} utils.timer.Timer.reduce
```

````

````{py:method} get_timing_metrics(reduction_op: typing.Union[str, dict[str, str]] = 'mean') -> dict[str, float | list[float]]
:canonical: utils.timer.Timer.get_timing_metrics

```{autodoc2-docstring} utils.timer.Timer.get_timing_metrics
```

````

````{py:method} reset(label: typing.Optional[str] = None) -> None
:canonical: utils.timer.Timer.reset

```{autodoc2-docstring} utils.timer.Timer.reset
```

````

`````
