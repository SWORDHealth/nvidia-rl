# {py:mod}`environments.math_environment`

```{py:module} environments.math_environment
```

```{autodoc2-docstring} environments.math_environment
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MathEnvConfig <environments.math_environment.MathEnvConfig>`
  -
* - {py:obj}`HFVerifyWorker <environments.math_environment.HFVerifyWorker>`
  - ```{autodoc2-docstring} environments.math_environment.HFVerifyWorker
    :summary:
    ```
* - {py:obj}`MathEnvironmentMetadata <environments.math_environment.MathEnvironmentMetadata>`
  -
* - {py:obj}`MathEnvironment <environments.math_environment.MathEnvironment>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_mute_output <environments.math_environment._mute_output>`
  - ```{autodoc2-docstring} environments.math_environment._mute_output
    :summary:
    ```
````

### API

`````{py:class} MathEnvConfig()
:canonical: environments.math_environment.MathEnvConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} num_workers
:canonical: environments.math_environment.MathEnvConfig.num_workers
:type: int
:value: >
   None

```{autodoc2-docstring} environments.math_environment.MathEnvConfig.num_workers
```

````

````{py:attribute} stop_strings
:canonical: environments.math_environment.MathEnvConfig.stop_strings
:type: typing.Optional[list[str]]
:value: >
   None

```{autodoc2-docstring} environments.math_environment.MathEnvConfig.stop_strings
```

````

`````

````{py:function} _mute_output()
:canonical: environments.math_environment._mute_output

```{autodoc2-docstring} environments.math_environment._mute_output
```
````

`````{py:class} HFVerifyWorker()
:canonical: environments.math_environment.HFVerifyWorker

```{autodoc2-docstring} environments.math_environment.HFVerifyWorker
```

```{rubric} Initialization
```

```{autodoc2-docstring} environments.math_environment.HFVerifyWorker.__init__
```

````{py:method} verify(pred_responses: list[str], ground_truths: list[str]) -> list[float]
:canonical: environments.math_environment.HFVerifyWorker.verify

```{autodoc2-docstring} environments.math_environment.HFVerifyWorker.verify
```

````

`````

`````{py:class} MathEnvironmentMetadata()
:canonical: environments.math_environment.MathEnvironmentMetadata

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} ground_truth
:canonical: environments.math_environment.MathEnvironmentMetadata.ground_truth
:type: str
:value: >
   None

```{autodoc2-docstring} environments.math_environment.MathEnvironmentMetadata.ground_truth
```

````

`````

`````{py:class} MathEnvironment(cfg: environments.math_environment.MathEnvConfig)
:canonical: environments.math_environment.MathEnvironment

Bases: {py:obj}`nemo_rl.environments.interfaces.EnvironmentInterface`

````{py:method} shutdown() -> None
:canonical: environments.math_environment.MathEnvironment.shutdown

```{autodoc2-docstring} environments.math_environment.MathEnvironment.shutdown
```

````

````{py:method} step(message_log_batch: list[list[dict[str, str]]], metadata: list[environments.math_environment.MathEnvironmentMetadata]) -> nemo_rl.environments.interfaces.EnvironmentReturn
:canonical: environments.math_environment.MathEnvironment.step

```{autodoc2-docstring} environments.math_environment.MathEnvironment.step
```

````

````{py:method} global_post_process_and_metrics(batch: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any]) -> tuple[nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any], dict[str, float | int]]
:canonical: environments.math_environment.MathEnvironment.global_post_process_and_metrics

```{autodoc2-docstring} environments.math_environment.MathEnvironment.global_post_process_and_metrics
```

````

`````
