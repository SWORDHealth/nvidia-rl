# {py:mod}`models.generation`

```{py:module} models.generation
```

```{autodoc2-docstring} models.generation
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

models.generation.interfaces
models.generation.vllm
models.generation.vllm_backend
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`configure_generation_config <models.generation.configure_generation_config>`
  - ```{autodoc2-docstring} models.generation.configure_generation_config
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <models.generation.TokenizerType>`
  - ```{autodoc2-docstring} models.generation.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: models.generation.TokenizerType
:value: >
   None

```{autodoc2-docstring} models.generation.TokenizerType
```

````

````{py:function} configure_generation_config(config: nemo_rl.models.generation.interfaces.GenerationConfig, tokenizer: models.generation.TokenizerType, is_eval=False) -> nemo_rl.models.generation.interfaces.GenerationConfig
:canonical: models.generation.configure_generation_config

```{autodoc2-docstring} models.generation.configure_generation_config
```
````
