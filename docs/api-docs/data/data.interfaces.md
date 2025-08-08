# {py:mod}`data.interfaces`

```{py:module} data.interfaces
```

```{autodoc2-docstring} data.interfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DatumSpec <data.interfaces.DatumSpec>`
  -
* - {py:obj}`DPODatumSpec <data.interfaces.DPODatumSpec>`
  -
* - {py:obj}`TaskDataSpec <data.interfaces.TaskDataSpec>`
  - ```{autodoc2-docstring} data.interfaces.TaskDataSpec
    :summary:
    ```
* - {py:obj}`TaskDataProcessFnCallable <data.interfaces.TaskDataProcessFnCallable>`
  - ```{autodoc2-docstring} data.interfaces.TaskDataProcessFnCallable
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LLMMessageLogType <data.interfaces.LLMMessageLogType>`
  - ```{autodoc2-docstring} data.interfaces.LLMMessageLogType
    :summary:
    ```
* - {py:obj}`FlatMessagesType <data.interfaces.FlatMessagesType>`
  - ```{autodoc2-docstring} data.interfaces.FlatMessagesType
    :summary:
    ```
* - {py:obj}`PathLike <data.interfaces.PathLike>`
  - ```{autodoc2-docstring} data.interfaces.PathLike
    :summary:
    ```
* - {py:obj}`TokenizerType <data.interfaces.TokenizerType>`
  - ```{autodoc2-docstring} data.interfaces.TokenizerType
    :summary:
    ```
````

### API

````{py:data} LLMMessageLogType
:canonical: data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} data.interfaces.LLMMessageLogType
```

````

````{py:data} FlatMessagesType
:canonical: data.interfaces.FlatMessagesType
:value: >
   None

```{autodoc2-docstring} data.interfaces.FlatMessagesType
```

````

````{py:data} PathLike
:canonical: data.interfaces.PathLike
:value: >
   None

```{autodoc2-docstring} data.interfaces.PathLike
```

````

````{py:data} TokenizerType
:canonical: data.interfaces.TokenizerType
:value: >
   None

```{autodoc2-docstring} data.interfaces.TokenizerType
```

````

`````{py:class} DatumSpec()
:canonical: data.interfaces.DatumSpec

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} message_log
:canonical: data.interfaces.DatumSpec.message_log
:type: data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.message_log
```

````

````{py:attribute} length
:canonical: data.interfaces.DatumSpec.length
:type: int
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.length
```

````

````{py:attribute} extra_env_info
:canonical: data.interfaces.DatumSpec.extra_env_info
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.extra_env_info
```

````

````{py:attribute} loss_multiplier
:canonical: data.interfaces.DatumSpec.loss_multiplier
:type: float
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.loss_multiplier
```

````

````{py:attribute} idx
:canonical: data.interfaces.DatumSpec.idx
:type: int
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.idx
```

````

````{py:attribute} task_name
:canonical: data.interfaces.DatumSpec.task_name
:type: typing.NotRequired[str]
:value: >
   'default'

```{autodoc2-docstring} data.interfaces.DatumSpec.task_name
```

````

````{py:attribute} stop_strings
:canonical: data.interfaces.DatumSpec.stop_strings
:type: typing.NotRequired[list[str]]
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.stop_strings
```

````

````{py:attribute} __extra__
:canonical: data.interfaces.DatumSpec.__extra__
:type: typing.NotRequired[typing.Any]
:value: >
   None

```{autodoc2-docstring} data.interfaces.DatumSpec.__extra__
```

````

`````

`````{py:class} DPODatumSpec()
:canonical: data.interfaces.DPODatumSpec

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} message_log_chosen
:canonical: data.interfaces.DPODatumSpec.message_log_chosen
:type: data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} data.interfaces.DPODatumSpec.message_log_chosen
```

````

````{py:attribute} message_log_rejected
:canonical: data.interfaces.DPODatumSpec.message_log_rejected
:type: data.interfaces.LLMMessageLogType
:value: >
   None

```{autodoc2-docstring} data.interfaces.DPODatumSpec.message_log_rejected
```

````

````{py:attribute} length_chosen
:canonical: data.interfaces.DPODatumSpec.length_chosen
:type: int
:value: >
   None

```{autodoc2-docstring} data.interfaces.DPODatumSpec.length_chosen
```

````

````{py:attribute} length_rejected
:canonical: data.interfaces.DPODatumSpec.length_rejected
:type: int
:value: >
   None

```{autodoc2-docstring} data.interfaces.DPODatumSpec.length_rejected
```

````

````{py:attribute} loss_multiplier
:canonical: data.interfaces.DPODatumSpec.loss_multiplier
:type: float
:value: >
   None

```{autodoc2-docstring} data.interfaces.DPODatumSpec.loss_multiplier
```

````

````{py:attribute} idx
:canonical: data.interfaces.DPODatumSpec.idx
:type: int
:value: >
   None

```{autodoc2-docstring} data.interfaces.DPODatumSpec.idx
```

````

`````

`````{py:class} TaskDataSpec
:canonical: data.interfaces.TaskDataSpec

```{autodoc2-docstring} data.interfaces.TaskDataSpec
```

````{py:attribute} task_name
:canonical: data.interfaces.TaskDataSpec.task_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} data.interfaces.TaskDataSpec.task_name
```

````

````{py:attribute} prompt_file
:canonical: data.interfaces.TaskDataSpec.prompt_file
:type: typing.Optional[data.interfaces.PathLike]
:value: >
   None

```{autodoc2-docstring} data.interfaces.TaskDataSpec.prompt_file
```

````

````{py:attribute} system_prompt_file
:canonical: data.interfaces.TaskDataSpec.system_prompt_file
:type: typing.Optional[data.interfaces.PathLike]
:value: >
   None

```{autodoc2-docstring} data.interfaces.TaskDataSpec.system_prompt_file
```

````

````{py:method} __post_init__() -> None
:canonical: data.interfaces.TaskDataSpec.__post_init__

```{autodoc2-docstring} data.interfaces.TaskDataSpec.__post_init__
```

````

````{py:method} copy_defaults(from_spec: data.interfaces.TaskDataSpec) -> None
:canonical: data.interfaces.TaskDataSpec.copy_defaults

```{autodoc2-docstring} data.interfaces.TaskDataSpec.copy_defaults
```

````

`````

`````{py:class} TaskDataProcessFnCallable
:canonical: data.interfaces.TaskDataProcessFnCallable

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} data.interfaces.TaskDataProcessFnCallable
```

````{py:method} __call__(datum_dict: dict[str, typing.Any], task_data_spec: data.interfaces.TaskDataSpec, tokenizer: data.interfaces.TokenizerType, max_seq_length: int, idx: int) -> data.interfaces.DatumSpec
:canonical: data.interfaces.TaskDataProcessFnCallable.__call__
:abstractmethod:

```{autodoc2-docstring} data.interfaces.TaskDataProcessFnCallable.__call__
```

````

`````
