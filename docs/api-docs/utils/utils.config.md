# {py:mod}`utils.config`

```{py:module} utils.config
```

```{autodoc2-docstring} utils.config
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`resolve_path <utils.config.resolve_path>`
  - ```{autodoc2-docstring} utils.config.resolve_path
    :summary:
    ```
* - {py:obj}`load_config_with_inheritance <utils.config.load_config_with_inheritance>`
  - ```{autodoc2-docstring} utils.config.load_config_with_inheritance
    :summary:
    ```
* - {py:obj}`load_config <utils.config.load_config>`
  - ```{autodoc2-docstring} utils.config.load_config
    :summary:
    ```
* - {py:obj}`parse_hydra_overrides <utils.config.parse_hydra_overrides>`
  - ```{autodoc2-docstring} utils.config.parse_hydra_overrides
    :summary:
    ```
````

### API

````{py:function} resolve_path(base_path: pathlib.Path, path: str) -> pathlib.Path
:canonical: utils.config.resolve_path

```{autodoc2-docstring} utils.config.resolve_path
```
````

````{py:function} load_config_with_inheritance(config_path: typing.Union[str, pathlib.Path], base_dir: typing.Optional[typing.Union[str, pathlib.Path]] = None) -> omegaconf.DictConfig
:canonical: utils.config.load_config_with_inheritance

```{autodoc2-docstring} utils.config.load_config_with_inheritance
```
````

````{py:function} load_config(config_path: typing.Union[str, pathlib.Path]) -> omegaconf.DictConfig
:canonical: utils.config.load_config

```{autodoc2-docstring} utils.config.load_config
```
````

````{py:exception} OverridesError()
:canonical: utils.config.OverridesError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} utils.config.OverridesError
```

```{rubric} Initialization
```

```{autodoc2-docstring} utils.config.OverridesError.__init__
```

````

````{py:function} parse_hydra_overrides(cfg: omegaconf.DictConfig, overrides: list[str]) -> omegaconf.DictConfig
:canonical: utils.config.parse_hydra_overrides

```{autodoc2-docstring} utils.config.parse_hydra_overrides
```
````
