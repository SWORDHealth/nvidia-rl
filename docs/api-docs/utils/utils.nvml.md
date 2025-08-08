# {py:mod}`utils.nvml`

```{py:module} utils.nvml
```

```{autodoc2-docstring} utils.nvml
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`nvml_context <utils.nvml.nvml_context>`
  - ```{autodoc2-docstring} utils.nvml.nvml_context
    :summary:
    ```
* - {py:obj}`device_id_to_physical_device_id <utils.nvml.device_id_to_physical_device_id>`
  - ```{autodoc2-docstring} utils.nvml.device_id_to_physical_device_id
    :summary:
    ```
* - {py:obj}`get_device_uuid <utils.nvml.get_device_uuid>`
  - ```{autodoc2-docstring} utils.nvml.get_device_uuid
    :summary:
    ```
* - {py:obj}`get_free_memory_bytes <utils.nvml.get_free_memory_bytes>`
  - ```{autodoc2-docstring} utils.nvml.get_free_memory_bytes
    :summary:
    ```
````

### API

````{py:function} nvml_context() -> typing.Generator[None, None, None]
:canonical: utils.nvml.nvml_context

```{autodoc2-docstring} utils.nvml.nvml_context
```
````

````{py:function} device_id_to_physical_device_id(device_id: int) -> int
:canonical: utils.nvml.device_id_to_physical_device_id

```{autodoc2-docstring} utils.nvml.device_id_to_physical_device_id
```
````

````{py:function} get_device_uuid(device_idx: int) -> str
:canonical: utils.nvml.get_device_uuid

```{autodoc2-docstring} utils.nvml.get_device_uuid
```
````

````{py:function} get_free_memory_bytes(device_idx: int) -> float
:canonical: utils.nvml.get_free_memory_bytes

```{autodoc2-docstring} utils.nvml.get_free_memory_bytes
```
````
