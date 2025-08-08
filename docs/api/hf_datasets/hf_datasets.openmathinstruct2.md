# {py:mod}`hf_datasets.openmathinstruct2`

```{py:module} hf_datasets.openmathinstruct2
```

```{autodoc2-docstring} hf_datasets.openmathinstruct2
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OpenMathInstruct2Dataset <hf_datasets.openmathinstruct2.OpenMathInstruct2Dataset>`
  - ```{autodoc2-docstring} hf_datasets.openmathinstruct2.OpenMathInstruct2Dataset
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`format_math <hf_datasets.openmathinstruct2.format_math>`
  - ```{autodoc2-docstring} hf_datasets.openmathinstruct2.format_math
    :summary:
    ```
* - {py:obj}`prepare_openinstructmath2_dataset <hf_datasets.openmathinstruct2.prepare_openinstructmath2_dataset>`
  - ```{autodoc2-docstring} hf_datasets.openmathinstruct2.prepare_openinstructmath2_dataset
    :summary:
    ```
````

### API

````{py:function} format_math(data: dict[str, str | float | int], output_key: str = 'expected_answer') -> dict[str, list[typing.Any] | str]
:canonical: hf_datasets.openmathinstruct2.format_math

```{autodoc2-docstring} hf_datasets.openmathinstruct2.format_math
```
````

````{py:function} prepare_openinstructmath2_dataset(split: str = 'train_1M', seed: int = 42, test_size: float = 0.05, output_key: str = 'expected_answer') -> dict[str, datasets.Dataset | None]
:canonical: hf_datasets.openmathinstruct2.prepare_openinstructmath2_dataset

```{autodoc2-docstring} hf_datasets.openmathinstruct2.prepare_openinstructmath2_dataset
```
````

````{py:class} OpenMathInstruct2Dataset(split: str = 'train_1M', seed: int = 42, test_size: float = 0.05, output_key: str = 'expected_answer', prompt_file: typing.Optional[str] = None)
:canonical: hf_datasets.openmathinstruct2.OpenMathInstruct2Dataset

```{autodoc2-docstring} hf_datasets.openmathinstruct2.OpenMathInstruct2Dataset
```

```{rubric} Initialization
```

```{autodoc2-docstring} hf_datasets.openmathinstruct2.OpenMathInstruct2Dataset.__init__
```

````
