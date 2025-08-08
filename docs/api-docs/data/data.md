# {py:mod}`data`

```{py:module} data
```

```{autodoc2-docstring} data
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

data.hf_datasets
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

data.datasets
data.interfaces
data.llm_message_utils
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DataConfig <data.DataConfig>`
  -
* - {py:obj}`MathDataConfig <data.MathDataConfig>`
  -
````

### API

`````{py:class} DataConfig()
:canonical: data.DataConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} max_input_seq_length
:canonical: data.DataConfig.max_input_seq_length
:type: int
:value: >
   None

```{autodoc2-docstring} data.DataConfig.max_input_seq_length
```

````

````{py:attribute} prompt_file
:canonical: data.DataConfig.prompt_file
:type: str
:value: >
   None

```{autodoc2-docstring} data.DataConfig.prompt_file
```

````

````{py:attribute} system_prompt_file
:canonical: data.DataConfig.system_prompt_file
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} data.DataConfig.system_prompt_file
```

````

````{py:attribute} dataset_name
:canonical: data.DataConfig.dataset_name
:type: str
:value: >
   None

```{autodoc2-docstring} data.DataConfig.dataset_name
```

````

````{py:attribute} val_dataset_name
:canonical: data.DataConfig.val_dataset_name
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} data.DataConfig.val_dataset_name
```

````

````{py:attribute} add_bos
:canonical: data.DataConfig.add_bos
:type: typing.Optional[bool]
:value: >
   None

```{autodoc2-docstring} data.DataConfig.add_bos
```

````

````{py:attribute} add_eos
:canonical: data.DataConfig.add_eos
:type: typing.Optional[bool]
:value: >
   None

```{autodoc2-docstring} data.DataConfig.add_eos
```

````

````{py:attribute} input_key
:canonical: data.DataConfig.input_key
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} data.DataConfig.input_key
```

````

````{py:attribute} output_key
:canonical: data.DataConfig.output_key
:type: typing.Optional[str]
:value: >
   None

```{autodoc2-docstring} data.DataConfig.output_key
```

````

`````

`````{py:class} MathDataConfig()
:canonical: data.MathDataConfig

Bases: {py:obj}`data.DataConfig`

````{py:attribute} problem_key
:canonical: data.MathDataConfig.problem_key
:type: str
:value: >
   None

```{autodoc2-docstring} data.MathDataConfig.problem_key
```

````

````{py:attribute} solution_key
:canonical: data.MathDataConfig.solution_key
:type: str
:value: >
   None

```{autodoc2-docstring} data.MathDataConfig.solution_key
```

````

`````
