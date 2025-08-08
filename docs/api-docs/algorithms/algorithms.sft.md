# {py:mod}`algorithms.sft`

```{py:module} algorithms.sft
```

```{autodoc2-docstring} algorithms.sft
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SFTSaveState <algorithms.sft.SFTSaveState>`
  -
* - {py:obj}`SFTConfig <algorithms.sft.SFTConfig>`
  -
* - {py:obj}`MasterConfig <algorithms.sft.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_default_sft_save_state <algorithms.sft._default_sft_save_state>`
  - ```{autodoc2-docstring} algorithms.sft._default_sft_save_state
    :summary:
    ```
* - {py:obj}`setup <algorithms.sft.setup>`
  - ```{autodoc2-docstring} algorithms.sft.setup
    :summary:
    ```
* - {py:obj}`validate <algorithms.sft.validate>`
  - ```{autodoc2-docstring} algorithms.sft.validate
    :summary:
    ```
* - {py:obj}`sft_train <algorithms.sft.sft_train>`
  - ```{autodoc2-docstring} algorithms.sft.sft_train
    :summary:
    ```
````

### API

`````{py:class} SFTSaveState()
:canonical: algorithms.sft.SFTSaveState

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} epoch
:canonical: algorithms.sft.SFTSaveState.epoch
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTSaveState.epoch
```

````

````{py:attribute} step
:canonical: algorithms.sft.SFTSaveState.step
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTSaveState.step
```

````

````{py:attribute} total_steps
:canonical: algorithms.sft.SFTSaveState.total_steps
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTSaveState.total_steps
```

````

````{py:attribute} val_loss
:canonical: algorithms.sft.SFTSaveState.val_loss
:type: float
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTSaveState.val_loss
```

````

````{py:attribute} consumed_samples
:canonical: algorithms.sft.SFTSaveState.consumed_samples
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTSaveState.consumed_samples
```

````

`````

````{py:function} _default_sft_save_state() -> algorithms.sft.SFTSaveState
:canonical: algorithms.sft._default_sft_save_state

```{autodoc2-docstring} algorithms.sft._default_sft_save_state
```
````

`````{py:class} SFTConfig()
:canonical: algorithms.sft.SFTConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} max_num_steps
:canonical: algorithms.sft.SFTConfig.max_num_steps
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.max_num_steps
```

````

````{py:attribute} max_num_epochs
:canonical: algorithms.sft.SFTConfig.max_num_epochs
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.max_num_epochs
```

````

````{py:attribute} val_period
:canonical: algorithms.sft.SFTConfig.val_period
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.val_period
```

````

````{py:attribute} val_batches
:canonical: algorithms.sft.SFTConfig.val_batches
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.val_batches
```

````

````{py:attribute} val_global_batch_size
:canonical: algorithms.sft.SFTConfig.val_global_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.val_global_batch_size
```

````

````{py:attribute} val_micro_batch_size
:canonical: algorithms.sft.SFTConfig.val_micro_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.val_micro_batch_size
```

````

````{py:attribute} val_at_start
:canonical: algorithms.sft.SFTConfig.val_at_start
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.val_at_start
```

````

````{py:attribute} seed
:canonical: algorithms.sft.SFTConfig.seed
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.sft.SFTConfig.seed
```

````

`````

`````{py:class} MasterConfig()
:canonical: algorithms.sft.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} policy
:canonical: algorithms.sft.MasterConfig.policy
:type: nemo_rl.models.policy.PolicyConfig
:value: >
   None

```{autodoc2-docstring} algorithms.sft.MasterConfig.policy
```

````

````{py:attribute} data
:canonical: algorithms.sft.MasterConfig.data
:type: nemo_rl.data.DataConfig
:value: >
   None

```{autodoc2-docstring} algorithms.sft.MasterConfig.data
```

````

````{py:attribute} sft
:canonical: algorithms.sft.MasterConfig.sft
:type: algorithms.sft.SFTConfig
:value: >
   None

```{autodoc2-docstring} algorithms.sft.MasterConfig.sft
```

````

````{py:attribute} logger
:canonical: algorithms.sft.MasterConfig.logger
:type: nemo_rl.utils.logger.LoggerConfig
:value: >
   None

```{autodoc2-docstring} algorithms.sft.MasterConfig.logger
```

````

````{py:attribute} cluster
:canonical: algorithms.sft.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} algorithms.sft.MasterConfig.cluster
```

````

````{py:attribute} checkpointing
:canonical: algorithms.sft.MasterConfig.checkpointing
:type: nemo_rl.utils.checkpoint.CheckpointingConfig
:value: >
   None

```{autodoc2-docstring} algorithms.sft.MasterConfig.checkpointing
```

````

`````

````{py:function} setup(master_config: algorithms.sft.MasterConfig, tokenizer: transformers.AutoTokenizer, train_dataset: nemo_rl.data.datasets.AllTaskProcessedDataset, val_dataset: nemo_rl.data.datasets.AllTaskProcessedDataset) -> tuple[nemo_rl.models.policy.lm_policy.Policy, nemo_rl.distributed.virtual_cluster.RayVirtualCluster, torchdata.stateful_dataloader.StatefulDataLoader, torchdata.stateful_dataloader.StatefulDataLoader, nemo_rl.algorithms.loss_functions.NLLLoss, algorithms.sft.MasterConfig, nemo_rl.utils.logger.Logger, nemo_rl.data.interfaces.TaskDataSpec, algorithms.sft.SFTSaveState]
:canonical: algorithms.sft.setup

```{autodoc2-docstring} algorithms.sft.setup
```
````

````{py:function} validate(policy: nemo_rl.models.policy.interfaces.PolicyInterface, val_dataloader: torchdata.stateful_dataloader.StatefulDataLoader, tokenizer, loss_fn, step: int, master_config: algorithms.sft.MasterConfig, sft_task_spec: nemo_rl.data.interfaces.TaskDataSpec, val_batches: int, val_batch_size: int, val_mbs: int)
:canonical: algorithms.sft.validate

```{autodoc2-docstring} algorithms.sft.validate
```
````

````{py:function} sft_train(policy, train_dataloader, val_dataloader, tokenizer, loss_fn, master_config, logger, sft_task_spec, checkpointer, sft_save_state)
:canonical: algorithms.sft.sft_train

```{autodoc2-docstring} algorithms.sft.sft_train
```
````
