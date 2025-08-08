# {py:mod}`distributed.virtual_cluster`

```{py:module} distributed.virtual_cluster
```

```{autodoc2-docstring} distributed.virtual_cluster
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ClusterConfig <distributed.virtual_cluster.ClusterConfig>`
  -
* - {py:obj}`PY_EXECUTABLES <distributed.virtual_cluster.PY_EXECUTABLES>`
  - ```{autodoc2-docstring} distributed.virtual_cluster.PY_EXECUTABLES
    :summary:
    ```
* - {py:obj}`RayVirtualCluster <distributed.virtual_cluster.RayVirtualCluster>`
  - ```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_node_ip_and_free_port <distributed.virtual_cluster._get_node_ip_and_free_port>`
  - ```{autodoc2-docstring} distributed.virtual_cluster._get_node_ip_and_free_port
    :summary:
    ```
* - {py:obj}`init_ray <distributed.virtual_cluster.init_ray>`
  - ```{autodoc2-docstring} distributed.virtual_cluster.init_ray
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <distributed.virtual_cluster.logger>`
  - ```{autodoc2-docstring} distributed.virtual_cluster.logger
    :summary:
    ```
* - {py:obj}`dir_path <distributed.virtual_cluster.dir_path>`
  - ```{autodoc2-docstring} distributed.virtual_cluster.dir_path
    :summary:
    ```
* - {py:obj}`git_root <distributed.virtual_cluster.git_root>`
  - ```{autodoc2-docstring} distributed.virtual_cluster.git_root
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: distributed.virtual_cluster.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} distributed.virtual_cluster.logger
```

````

`````{py:class} ClusterConfig()
:canonical: distributed.virtual_cluster.ClusterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} gpus_per_node
:canonical: distributed.virtual_cluster.ClusterConfig.gpus_per_node
:type: int
:value: >
   None

```{autodoc2-docstring} distributed.virtual_cluster.ClusterConfig.gpus_per_node
```

````

````{py:attribute} num_nodes
:canonical: distributed.virtual_cluster.ClusterConfig.num_nodes
:type: int
:value: >
   None

```{autodoc2-docstring} distributed.virtual_cluster.ClusterConfig.num_nodes
```

````

`````

````{py:data} dir_path
:canonical: distributed.virtual_cluster.dir_path
:value: >
   'dirname(...)'

```{autodoc2-docstring} distributed.virtual_cluster.dir_path
```

````

````{py:data} git_root
:canonical: distributed.virtual_cluster.git_root
:value: >
   'abspath(...)'

```{autodoc2-docstring} distributed.virtual_cluster.git_root
```

````

`````{py:class} PY_EXECUTABLES
:canonical: distributed.virtual_cluster.PY_EXECUTABLES

```{autodoc2-docstring} distributed.virtual_cluster.PY_EXECUTABLES
```

````{py:attribute} SYSTEM
:canonical: distributed.virtual_cluster.PY_EXECUTABLES.SYSTEM
:value: >
   None

```{autodoc2-docstring} distributed.virtual_cluster.PY_EXECUTABLES.SYSTEM
```

````

````{py:attribute} BASE
:canonical: distributed.virtual_cluster.PY_EXECUTABLES.BASE
:value: >
   'uv run --locked'

```{autodoc2-docstring} distributed.virtual_cluster.PY_EXECUTABLES.BASE
```

````

````{py:attribute} VLLM
:canonical: distributed.virtual_cluster.PY_EXECUTABLES.VLLM
:value: >
   'uv run --locked --extra vllm'

```{autodoc2-docstring} distributed.virtual_cluster.PY_EXECUTABLES.VLLM
```

````

````{py:attribute} MCORE
:canonical: distributed.virtual_cluster.PY_EXECUTABLES.MCORE
:value: >
   'uv run --reinstall --extra mcore'

```{autodoc2-docstring} distributed.virtual_cluster.PY_EXECUTABLES.MCORE
```

````

`````

````{py:function} _get_node_ip_and_free_port() -> tuple[str, int]
:canonical: distributed.virtual_cluster._get_node_ip_and_free_port

```{autodoc2-docstring} distributed.virtual_cluster._get_node_ip_and_free_port
```
````

````{py:function} init_ray(log_dir: typing.Optional[str] = None) -> None
:canonical: distributed.virtual_cluster.init_ray

```{autodoc2-docstring} distributed.virtual_cluster.init_ray
```
````

````{py:exception} ResourceInsufficientError()
:canonical: distributed.virtual_cluster.ResourceInsufficientError

Bases: {py:obj}`Exception`

```{autodoc2-docstring} distributed.virtual_cluster.ResourceInsufficientError
```

```{rubric} Initialization
```

```{autodoc2-docstring} distributed.virtual_cluster.ResourceInsufficientError.__init__
```

````

`````{py:class} RayVirtualCluster(bundle_ct_per_node_list: list[int], use_gpus: bool = True, max_colocated_worker_groups: int = 1, num_gpus_per_node: int = 8, name: str = '', placement_group_strategy: str = 'SPREAD')
:canonical: distributed.virtual_cluster.RayVirtualCluster

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster
```

```{rubric} Initialization
```

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.__init__
```

````{py:method} _init_placement_groups(strategy: str | None = None, use_unified_pg: bool | None = None) -> list[ray.util.placement_group.PlacementGroup]
:canonical: distributed.virtual_cluster.RayVirtualCluster._init_placement_groups

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster._init_placement_groups
```

````

````{py:method} _create_placement_groups_internal(strategy: str, use_unified_pg: bool = False) -> list[ray.util.placement_group.PlacementGroup]
:canonical: distributed.virtual_cluster.RayVirtualCluster._create_placement_groups_internal

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster._create_placement_groups_internal
```

````

````{py:method} get_placement_groups() -> list[ray.util.placement_group.PlacementGroup]
:canonical: distributed.virtual_cluster.RayVirtualCluster.get_placement_groups

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.get_placement_groups
```

````

````{py:method} world_size() -> int
:canonical: distributed.virtual_cluster.RayVirtualCluster.world_size

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.world_size
```

````

````{py:method} node_count() -> int
:canonical: distributed.virtual_cluster.RayVirtualCluster.node_count

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.node_count
```

````

````{py:method} get_master_address_and_port() -> tuple[str, int]
:canonical: distributed.virtual_cluster.RayVirtualCluster.get_master_address_and_port

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.get_master_address_and_port
```

````

````{py:method} shutdown() -> bool
:canonical: distributed.virtual_cluster.RayVirtualCluster.shutdown

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.shutdown
```

````

````{py:method} __del__() -> None
:canonical: distributed.virtual_cluster.RayVirtualCluster.__del__

```{autodoc2-docstring} distributed.virtual_cluster.RayVirtualCluster.__del__
```

````

`````
