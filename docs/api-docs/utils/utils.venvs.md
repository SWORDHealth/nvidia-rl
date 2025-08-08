# {py:mod}`utils.venvs`

```{py:module} utils.venvs
```

```{autodoc2-docstring} utils.venvs
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_local_venv <utils.venvs.create_local_venv>`
  - ```{autodoc2-docstring} utils.venvs.create_local_venv
    :summary:
    ```
* - {py:obj}`_env_builder <utils.venvs._env_builder>`
  - ```{autodoc2-docstring} utils.venvs._env_builder
    :summary:
    ```
* - {py:obj}`create_local_venv_on_each_node <utils.venvs.create_local_venv_on_each_node>`
  - ```{autodoc2-docstring} utils.venvs.create_local_venv_on_each_node
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`dir_path <utils.venvs.dir_path>`
  - ```{autodoc2-docstring} utils.venvs.dir_path
    :summary:
    ```
* - {py:obj}`git_root <utils.venvs.git_root>`
  - ```{autodoc2-docstring} utils.venvs.git_root
    :summary:
    ```
* - {py:obj}`DEFAULT_VENV_DIR <utils.venvs.DEFAULT_VENV_DIR>`
  - ```{autodoc2-docstring} utils.venvs.DEFAULT_VENV_DIR
    :summary:
    ```
* - {py:obj}`logger <utils.venvs.logger>`
  - ```{autodoc2-docstring} utils.venvs.logger
    :summary:
    ```
````

### API

````{py:data} dir_path
:canonical: utils.venvs.dir_path
:value: >
   'dirname(...)'

```{autodoc2-docstring} utils.venvs.dir_path
```

````

````{py:data} git_root
:canonical: utils.venvs.git_root
:value: >
   'abspath(...)'

```{autodoc2-docstring} utils.venvs.git_root
```

````

````{py:data} DEFAULT_VENV_DIR
:canonical: utils.venvs.DEFAULT_VENV_DIR
:value: >
   'join(...)'

```{autodoc2-docstring} utils.venvs.DEFAULT_VENV_DIR
```

````

````{py:data} logger
:canonical: utils.venvs.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} utils.venvs.logger
```

````

````{py:function} create_local_venv(py_executable: str, venv_name: str, force_rebuild: bool = False) -> str
:canonical: utils.venvs.create_local_venv

```{autodoc2-docstring} utils.venvs.create_local_venv
```
````

````{py:function} _env_builder(py_executable: str, venv_name: str, node_idx: int, force_rebuild: bool = False)
:canonical: utils.venvs._env_builder

```{autodoc2-docstring} utils.venvs._env_builder
```
````

````{py:function} create_local_venv_on_each_node(py_executable: str, venv_name: str)
:canonical: utils.venvs.create_local_venv_on_each_node

```{autodoc2-docstring} utils.venvs.create_local_venv_on_each_node
```
````
