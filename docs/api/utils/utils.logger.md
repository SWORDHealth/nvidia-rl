# {py:mod}`utils.logger`

```{py:module} utils.logger
```

```{autodoc2-docstring} utils.logger
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WandbConfig <utils.logger.WandbConfig>`
  -
* - {py:obj}`TensorboardConfig <utils.logger.TensorboardConfig>`
  -
* - {py:obj}`GPUMonitoringConfig <utils.logger.GPUMonitoringConfig>`
  -
* - {py:obj}`LoggerConfig <utils.logger.LoggerConfig>`
  -
* - {py:obj}`LoggerInterface <utils.logger.LoggerInterface>`
  - ```{autodoc2-docstring} utils.logger.LoggerInterface
    :summary:
    ```
* - {py:obj}`TensorboardLogger <utils.logger.TensorboardLogger>`
  - ```{autodoc2-docstring} utils.logger.TensorboardLogger
    :summary:
    ```
* - {py:obj}`WandbLogger <utils.logger.WandbLogger>`
  - ```{autodoc2-docstring} utils.logger.WandbLogger
    :summary:
    ```
* - {py:obj}`GpuMetricSnapshot <utils.logger.GpuMetricSnapshot>`
  -
* - {py:obj}`RayGpuMonitorLogger <utils.logger.RayGpuMonitorLogger>`
  - ```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger
    :summary:
    ```
* - {py:obj}`Logger <utils.logger.Logger>`
  - ```{autodoc2-docstring} utils.logger.Logger
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`flatten_dict <utils.logger.flatten_dict>`
  - ```{autodoc2-docstring} utils.logger.flatten_dict
    :summary:
    ```
* - {py:obj}`configure_rich_logging <utils.logger.configure_rich_logging>`
  - ```{autodoc2-docstring} utils.logger.configure_rich_logging
    :summary:
    ```
* - {py:obj}`print_message_log_samples <utils.logger.print_message_log_samples>`
  - ```{autodoc2-docstring} utils.logger.print_message_log_samples
    :summary:
    ```
* - {py:obj}`get_next_experiment_dir <utils.logger.get_next_experiment_dir>`
  - ```{autodoc2-docstring} utils.logger.get_next_experiment_dir
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_rich_logging_configured <utils.logger._rich_logging_configured>`
  - ```{autodoc2-docstring} utils.logger._rich_logging_configured
    :summary:
    ```
````

### API

````{py:data} _rich_logging_configured
:canonical: utils.logger._rich_logging_configured
:value: >
   False

```{autodoc2-docstring} utils.logger._rich_logging_configured
```

````

`````{py:class} WandbConfig()
:canonical: utils.logger.WandbConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} project
:canonical: utils.logger.WandbConfig.project
:type: str
:value: >
   None

```{autodoc2-docstring} utils.logger.WandbConfig.project
```

````

````{py:attribute} name
:canonical: utils.logger.WandbConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} utils.logger.WandbConfig.name
```

````

`````

`````{py:class} TensorboardConfig()
:canonical: utils.logger.TensorboardConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} log_dir
:canonical: utils.logger.TensorboardConfig.log_dir
:type: str
:value: >
   None

```{autodoc2-docstring} utils.logger.TensorboardConfig.log_dir
```

````

`````

`````{py:class} GPUMonitoringConfig()
:canonical: utils.logger.GPUMonitoringConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} collection_interval
:canonical: utils.logger.GPUMonitoringConfig.collection_interval
:type: int | float
:value: >
   None

```{autodoc2-docstring} utils.logger.GPUMonitoringConfig.collection_interval
```

````

````{py:attribute} flush_interval
:canonical: utils.logger.GPUMonitoringConfig.flush_interval
:type: int | float
:value: >
   None

```{autodoc2-docstring} utils.logger.GPUMonitoringConfig.flush_interval
```

````

`````

`````{py:class} LoggerConfig()
:canonical: utils.logger.LoggerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} log_dir
:canonical: utils.logger.LoggerConfig.log_dir
:type: str
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.log_dir
```

````

````{py:attribute} wandb_enabled
:canonical: utils.logger.LoggerConfig.wandb_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.wandb_enabled
```

````

````{py:attribute} tensorboard_enabled
:canonical: utils.logger.LoggerConfig.tensorboard_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.tensorboard_enabled
```

````

````{py:attribute} wandb
:canonical: utils.logger.LoggerConfig.wandb
:type: utils.logger.WandbConfig
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.wandb
```

````

````{py:attribute} tensorboard
:canonical: utils.logger.LoggerConfig.tensorboard
:type: utils.logger.TensorboardConfig
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.tensorboard
```

````

````{py:attribute} monitor_gpus
:canonical: utils.logger.LoggerConfig.monitor_gpus
:type: bool
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.monitor_gpus
```

````

````{py:attribute} gpu_monitoring
:canonical: utils.logger.LoggerConfig.gpu_monitoring
:type: utils.logger.GPUMonitoringConfig
:value: >
   None

```{autodoc2-docstring} utils.logger.LoggerConfig.gpu_monitoring
```

````

`````

`````{py:class} LoggerInterface
:canonical: utils.logger.LoggerInterface

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} utils.logger.LoggerInterface
```

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: utils.logger.LoggerInterface.log_metrics
:abstractmethod:

```{autodoc2-docstring} utils.logger.LoggerInterface.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: utils.logger.LoggerInterface.log_hyperparams
:abstractmethod:

```{autodoc2-docstring} utils.logger.LoggerInterface.log_hyperparams
```

````

`````

`````{py:class} TensorboardLogger(cfg: utils.logger.TensorboardConfig, log_dir: typing.Optional[str] = None)
:canonical: utils.logger.TensorboardLogger

Bases: {py:obj}`utils.logger.LoggerInterface`

```{autodoc2-docstring} utils.logger.TensorboardLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.logger.TensorboardLogger.__init__
```

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: utils.logger.TensorboardLogger.log_metrics

```{autodoc2-docstring} utils.logger.TensorboardLogger.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: utils.logger.TensorboardLogger.log_hyperparams

```{autodoc2-docstring} utils.logger.TensorboardLogger.log_hyperparams
```

````

````{py:method} log_plot(figure: matplotlib.pyplot.Figure, step: int, name: str) -> None
:canonical: utils.logger.TensorboardLogger.log_plot

```{autodoc2-docstring} utils.logger.TensorboardLogger.log_plot
```

````

`````

`````{py:class} WandbLogger(cfg: utils.logger.WandbConfig, log_dir: typing.Optional[str] = None)
:canonical: utils.logger.WandbLogger

Bases: {py:obj}`utils.logger.LoggerInterface`

```{autodoc2-docstring} utils.logger.WandbLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.logger.WandbLogger.__init__
```

````{py:method} _log_diffs()
:canonical: utils.logger.WandbLogger._log_diffs

```{autodoc2-docstring} utils.logger.WandbLogger._log_diffs
```

````

````{py:method} _log_code()
:canonical: utils.logger.WandbLogger._log_code

```{autodoc2-docstring} utils.logger.WandbLogger._log_code
```

````

````{py:method} define_metric(name: str, step_metric: typing.Optional[str] = None) -> None
:canonical: utils.logger.WandbLogger.define_metric

```{autodoc2-docstring} utils.logger.WandbLogger.define_metric
```

````

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: utils.logger.WandbLogger.log_metrics

```{autodoc2-docstring} utils.logger.WandbLogger.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: utils.logger.WandbLogger.log_hyperparams

```{autodoc2-docstring} utils.logger.WandbLogger.log_hyperparams
```

````

````{py:method} log_plot(figure: matplotlib.pyplot.Figure, step: int, name: str) -> None
:canonical: utils.logger.WandbLogger.log_plot

```{autodoc2-docstring} utils.logger.WandbLogger.log_plot
```

````

`````

`````{py:class} GpuMetricSnapshot()
:canonical: utils.logger.GpuMetricSnapshot

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} step
:canonical: utils.logger.GpuMetricSnapshot.step
:type: int
:value: >
   None

```{autodoc2-docstring} utils.logger.GpuMetricSnapshot.step
```

````

````{py:attribute} metrics
:canonical: utils.logger.GpuMetricSnapshot.metrics
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} utils.logger.GpuMetricSnapshot.metrics
```

````

`````

`````{py:class} RayGpuMonitorLogger(collection_interval: int | float, flush_interval: int | float, metric_prefix: str, step_metric: str, parent_logger: typing.Optional[utils.logger.Logger] = None)
:canonical: utils.logger.RayGpuMonitorLogger

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger.__init__
```

````{py:method} start() -> None
:canonical: utils.logger.RayGpuMonitorLogger.start

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger.start
```

````

````{py:method} stop() -> None
:canonical: utils.logger.RayGpuMonitorLogger.stop

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger.stop
```

````

````{py:method} _collection_loop() -> None
:canonical: utils.logger.RayGpuMonitorLogger._collection_loop

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._collection_loop
```

````

````{py:method} _parse_metric(sample: prometheus_client.samples.Sample, node_idx: int) -> dict[str, typing.Any]
:canonical: utils.logger.RayGpuMonitorLogger._parse_metric

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._parse_metric
```

````

````{py:method} _parse_gpu_sku(sample: prometheus_client.samples.Sample, node_idx: int) -> dict[str, str]
:canonical: utils.logger.RayGpuMonitorLogger._parse_gpu_sku

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._parse_gpu_sku
```

````

````{py:method} _collect_gpu_sku() -> dict[str, str]
:canonical: utils.logger.RayGpuMonitorLogger._collect_gpu_sku

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._collect_gpu_sku
```

````

````{py:method} _collect_metrics() -> dict[str, typing.Any]
:canonical: utils.logger.RayGpuMonitorLogger._collect_metrics

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._collect_metrics
```

````

````{py:method} _collect(metrics: bool = False, sku: bool = False) -> dict[str, typing.Any]
:canonical: utils.logger.RayGpuMonitorLogger._collect

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._collect
```

````

````{py:method} _fetch_and_parse_metrics(node_idx: int, metric_address: str, parser_fn: typing.Callable)
:canonical: utils.logger.RayGpuMonitorLogger._fetch_and_parse_metrics

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger._fetch_and_parse_metrics
```

````

````{py:method} flush() -> None
:canonical: utils.logger.RayGpuMonitorLogger.flush

```{autodoc2-docstring} utils.logger.RayGpuMonitorLogger.flush
```

````

`````

`````{py:class} Logger(cfg: utils.logger.LoggerConfig)
:canonical: utils.logger.Logger

Bases: {py:obj}`utils.logger.LoggerInterface`

```{autodoc2-docstring} utils.logger.Logger
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.logger.Logger.__init__
```

````{py:method} log_metrics(metrics: dict[str, typing.Any], step: int, prefix: typing.Optional[str] = '', step_metric: typing.Optional[str] = None) -> None
:canonical: utils.logger.Logger.log_metrics

```{autodoc2-docstring} utils.logger.Logger.log_metrics
```

````

````{py:method} log_hyperparams(params: typing.Mapping[str, typing.Any]) -> None
:canonical: utils.logger.Logger.log_hyperparams

```{autodoc2-docstring} utils.logger.Logger.log_hyperparams
```

````

````{py:method} log_batched_dict_as_jsonl(to_log: nemo_rl.distributed.batched_data_dict.BatchedDataDict[typing.Any] | dict[str, typing.Any], filename: str) -> None
:canonical: utils.logger.Logger.log_batched_dict_as_jsonl

```{autodoc2-docstring} utils.logger.Logger.log_batched_dict_as_jsonl
```

````

````{py:method} log_plot_token_mult_prob_error(data: dict[str, typing.Any], step: int, name: str) -> None
:canonical: utils.logger.Logger.log_plot_token_mult_prob_error

```{autodoc2-docstring} utils.logger.Logger.log_plot_token_mult_prob_error
```

````

````{py:method} __del__() -> None
:canonical: utils.logger.Logger.__del__

```{autodoc2-docstring} utils.logger.Logger.__del__
```

````

`````

````{py:function} flatten_dict(d: typing.Mapping[str, typing.Any], sep: str = '.') -> dict[str, typing.Any]
:canonical: utils.logger.flatten_dict

```{autodoc2-docstring} utils.logger.flatten_dict
```
````

````{py:function} configure_rich_logging(level: str = 'INFO', show_time: bool = True, show_path: bool = True) -> None
:canonical: utils.logger.configure_rich_logging

```{autodoc2-docstring} utils.logger.configure_rich_logging
```
````

````{py:function} print_message_log_samples(message_logs: list[nemo_rl.data.interfaces.LLMMessageLogType], rewards: list[float], num_samples: int = 5, step: int = 0) -> None
:canonical: utils.logger.print_message_log_samples

```{autodoc2-docstring} utils.logger.print_message_log_samples
```
````

````{py:function} get_next_experiment_dir(base_log_dir: str) -> str
:canonical: utils.logger.get_next_experiment_dir

```{autodoc2-docstring} utils.logger.get_next_experiment_dir
```
````
