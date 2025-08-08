# {py:mod}`utils.nsys`

```{py:module} utils.nsys
```

```{autodoc2-docstring} utils.nsys
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProfilablePolicy <utils.nsys.ProfilablePolicy>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`maybe_gpu_profile_step <utils.nsys.maybe_gpu_profile_step>`
  - ```{autodoc2-docstring} utils.nsys.maybe_gpu_profile_step
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NRL_NSYS_WORKER_PATTERNS <utils.nsys.NRL_NSYS_WORKER_PATTERNS>`
  - ```{autodoc2-docstring} utils.nsys.NRL_NSYS_WORKER_PATTERNS
    :summary:
    ```
* - {py:obj}`NRL_NSYS_PROFILE_STEP_RANGE <utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE>`
  - ```{autodoc2-docstring} utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE
    :summary:
    ```
````

### API

````{py:data} NRL_NSYS_WORKER_PATTERNS
:canonical: utils.nsys.NRL_NSYS_WORKER_PATTERNS
:value: >
   'get(...)'

```{autodoc2-docstring} utils.nsys.NRL_NSYS_WORKER_PATTERNS
```

````

````{py:data} NRL_NSYS_PROFILE_STEP_RANGE
:canonical: utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE
:value: >
   'get(...)'

```{autodoc2-docstring} utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE
```

````

`````{py:class} ProfilablePolicy
:canonical: utils.nsys.ProfilablePolicy

Bases: {py:obj}`typing.Protocol`

````{py:method} start_gpu_profiling() -> None
:canonical: utils.nsys.ProfilablePolicy.start_gpu_profiling

```{autodoc2-docstring} utils.nsys.ProfilablePolicy.start_gpu_profiling
```

````

````{py:method} stop_gpu_profiling() -> None
:canonical: utils.nsys.ProfilablePolicy.stop_gpu_profiling

```{autodoc2-docstring} utils.nsys.ProfilablePolicy.stop_gpu_profiling
```

````

`````

````{py:function} maybe_gpu_profile_step(policy: utils.nsys.ProfilablePolicy, step: int)
:canonical: utils.nsys.maybe_gpu_profile_step

```{autodoc2-docstring} utils.nsys.maybe_gpu_profile_step
```
````
