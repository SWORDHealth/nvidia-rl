# {py:mod}`algorithms.grpo`

```{py:module} algorithms.grpo
```

```{autodoc2-docstring} algorithms.grpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GRPOConfig <algorithms.grpo.GRPOConfig>`
  -
* - {py:obj}`GRPOSaveState <algorithms.grpo.GRPOSaveState>`
  -
* - {py:obj}`GRPOLoggerConfig <algorithms.grpo.GRPOLoggerConfig>`
  -
* - {py:obj}`MasterConfig <algorithms.grpo.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_default_grpo_save_state <algorithms.grpo._default_grpo_save_state>`
  - ```{autodoc2-docstring} algorithms.grpo._default_grpo_save_state
    :summary:
    ```
* - {py:obj}`setup <algorithms.grpo.setup>`
  - ```{autodoc2-docstring} algorithms.grpo.setup
    :summary:
    ```
* - {py:obj}`_should_use_async_rollouts <algorithms.grpo._should_use_async_rollouts>`
  - ```{autodoc2-docstring} algorithms.grpo._should_use_async_rollouts
    :summary:
    ```
* - {py:obj}`refit_policy_generation <algorithms.grpo.refit_policy_generation>`
  - ```{autodoc2-docstring} algorithms.grpo.refit_policy_generation
    :summary:
    ```
* - {py:obj}`grpo_train <algorithms.grpo.grpo_train>`
  - ```{autodoc2-docstring} algorithms.grpo.grpo_train
    :summary:
    ```
* - {py:obj}`validate <algorithms.grpo.validate>`
  - ```{autodoc2-docstring} algorithms.grpo.validate
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TokenizerType <algorithms.grpo.TokenizerType>`
  - ```{autodoc2-docstring} algorithms.grpo.TokenizerType
    :summary:
    ```
````

### API

````{py:data} TokenizerType
:canonical: algorithms.grpo.TokenizerType
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} algorithms.grpo.TokenizerType
```

````

`````{py:class} GRPOConfig()
:canonical: algorithms.grpo.GRPOConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} num_prompts_per_step
:canonical: algorithms.grpo.GRPOConfig.num_prompts_per_step
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.num_prompts_per_step
```

````

````{py:attribute} num_generations_per_prompt
:canonical: algorithms.grpo.GRPOConfig.num_generations_per_prompt
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.num_generations_per_prompt
```

````

````{py:attribute} max_num_steps
:canonical: algorithms.grpo.GRPOConfig.max_num_steps
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.max_num_steps
```

````

````{py:attribute} max_rollout_turns
:canonical: algorithms.grpo.GRPOConfig.max_rollout_turns
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.max_rollout_turns
```

````

````{py:attribute} normalize_rewards
:canonical: algorithms.grpo.GRPOConfig.normalize_rewards
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.normalize_rewards
```

````

````{py:attribute} use_leave_one_out_baseline
:canonical: algorithms.grpo.GRPOConfig.use_leave_one_out_baseline
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.use_leave_one_out_baseline
```

````

````{py:attribute} val_period
:canonical: algorithms.grpo.GRPOConfig.val_period
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.val_period
```

````

````{py:attribute} val_batch_size
:canonical: algorithms.grpo.GRPOConfig.val_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.val_batch_size
```

````

````{py:attribute} val_at_start
:canonical: algorithms.grpo.GRPOConfig.val_at_start
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.val_at_start
```

````

````{py:attribute} max_val_samples
:canonical: algorithms.grpo.GRPOConfig.max_val_samples
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.max_val_samples
```

````

````{py:attribute} checkpoint_dir
:canonical: algorithms.grpo.GRPOConfig.checkpoint_dir
:type: str
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOConfig.checkpoint_dir
```

````

`````

`````{py:class} GRPOSaveState()
:canonical: algorithms.grpo.GRPOSaveState

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} step
:canonical: algorithms.grpo.GRPOSaveState.step
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOSaveState.step
```

````

````{py:attribute} val_reward
:canonical: algorithms.grpo.GRPOSaveState.val_reward
:type: float
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOSaveState.val_reward
```

````

````{py:attribute} consumed_samples
:canonical: algorithms.grpo.GRPOSaveState.consumed_samples
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOSaveState.consumed_samples
```

````

`````

````{py:function} _default_grpo_save_state() -> algorithms.grpo.GRPOSaveState
:canonical: algorithms.grpo._default_grpo_save_state

```{autodoc2-docstring} algorithms.grpo._default_grpo_save_state
```
````

`````{py:class} GRPOLoggerConfig()
:canonical: algorithms.grpo.GRPOLoggerConfig

Bases: {py:obj}`nemo_rl.utils.logger.LoggerConfig`

````{py:attribute} num_val_samples_to_print
:canonical: algorithms.grpo.GRPOLoggerConfig.num_val_samples_to_print
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.GRPOLoggerConfig.num_val_samples_to_print
```

````

`````

`````{py:class} MasterConfig()
:canonical: algorithms.grpo.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} policy
:canonical: algorithms.grpo.MasterConfig.policy
:type: nemo_rl.models.policy.PolicyConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.policy
```

````

````{py:attribute} loss_fn
:canonical: algorithms.grpo.MasterConfig.loss_fn
:type: nemo_rl.algorithms.loss_functions.ClippedPGLossConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.loss_fn
```

````

````{py:attribute} env
:canonical: algorithms.grpo.MasterConfig.env
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.env
```

````

````{py:attribute} data
:canonical: algorithms.grpo.MasterConfig.data
:type: nemo_rl.data.DataConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.data
```

````

````{py:attribute} grpo
:canonical: algorithms.grpo.MasterConfig.grpo
:type: algorithms.grpo.GRPOConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.grpo
```

````

````{py:attribute} logger
:canonical: algorithms.grpo.MasterConfig.logger
:type: algorithms.grpo.GRPOLoggerConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.logger
```

````

````{py:attribute} cluster
:canonical: algorithms.grpo.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.cluster
```

````

````{py:attribute} checkpointing
:canonical: algorithms.grpo.MasterConfig.checkpointing
:type: nemo_rl.utils.checkpoint.CheckpointingConfig
:value: >
   None

```{autodoc2-docstring} algorithms.grpo.MasterConfig.checkpointing
```

````

`````

````{py:function} setup(master_config: algorithms.grpo.MasterConfig, tokenizer: algorithms.grpo.TokenizerType, dataset: nemo_rl.data.datasets.AllTaskProcessedDataset, val_dataset: typing.Optional[nemo_rl.data.datasets.AllTaskProcessedDataset]) -> tuple[nemo_rl.models.policy.interfaces.ColocatablePolicyInterface, typing.Optional[nemo_rl.models.generation.interfaces.GenerationInterface], typing.Tuple[nemo_rl.distributed.virtual_cluster.RayVirtualCluster, nemo_rl.distributed.virtual_cluster.RayVirtualCluster], torchdata.stateful_dataloader.StatefulDataLoader, typing.Optional[torchdata.stateful_dataloader.StatefulDataLoader], nemo_rl.algorithms.loss_functions.ClippedPGLossFn, nemo_rl.utils.logger.Logger, nemo_rl.utils.checkpoint.CheckpointManager, algorithms.grpo.GRPOSaveState, algorithms.grpo.MasterConfig]
:canonical: algorithms.grpo.setup

```{autodoc2-docstring} algorithms.grpo.setup
```
````

````{py:function} _should_use_async_rollouts(master_config: algorithms.grpo.MasterConfig) -> bool
:canonical: algorithms.grpo._should_use_async_rollouts

```{autodoc2-docstring} algorithms.grpo._should_use_async_rollouts
```
````

````{py:function} refit_policy_generation(policy: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface, policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, colocated_inference: bool, _refit_buffer_size_gb: typing.Optional[int] = None) -> None
:canonical: algorithms.grpo.refit_policy_generation

```{autodoc2-docstring} algorithms.grpo.refit_policy_generation
```
````

````{py:function} grpo_train(policy: nemo_rl.models.policy.interfaces.ColocatablePolicyInterface, policy_generation: typing.Optional[nemo_rl.models.generation.interfaces.GenerationInterface], dataloader: torchdata.stateful_dataloader.StatefulDataLoader, val_dataloader: typing.Optional[torchdata.stateful_dataloader.StatefulDataLoader], tokenizer: algorithms.grpo.TokenizerType, loss_fn: nemo_rl.algorithms.interfaces.LossFunction, task_to_env: dict[str, nemo_rl.environments.interfaces.EnvironmentInterface], val_task_to_env: typing.Optional[dict[str, nemo_rl.environments.interfaces.EnvironmentInterface]], logger: nemo_rl.utils.logger.Logger, checkpointer: nemo_rl.utils.checkpoint.CheckpointManager, grpo_save_state: algorithms.grpo.GRPOSaveState, master_config: algorithms.grpo.MasterConfig) -> None
:canonical: algorithms.grpo.grpo_train

```{autodoc2-docstring} algorithms.grpo.grpo_train
```
````

````{py:function} validate(policy_generation: nemo_rl.models.generation.interfaces.GenerationInterface, val_dataloader: typing.Optional[torchdata.stateful_dataloader.StatefulDataLoader], tokenizer, val_task_to_env: typing.Optional[dict[str, nemo_rl.environments.interfaces.EnvironmentInterface]], step: int, master_config: algorithms.grpo.MasterConfig) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]
:canonical: algorithms.grpo.validate

```{autodoc2-docstring} algorithms.grpo.validate
```
````
