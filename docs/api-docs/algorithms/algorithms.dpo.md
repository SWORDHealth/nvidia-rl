# {py:mod}`algorithms.dpo`

```{py:module} algorithms.dpo
```

```{autodoc2-docstring} algorithms.dpo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DPOSaveState <algorithms.dpo.DPOSaveState>`
  -
* - {py:obj}`DPOConfig <algorithms.dpo.DPOConfig>`
  -
* - {py:obj}`MasterConfig <algorithms.dpo.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_default_dpo_save_state <algorithms.dpo._default_dpo_save_state>`
  - ```{autodoc2-docstring} algorithms.dpo._default_dpo_save_state
    :summary:
    ```
* - {py:obj}`setup <algorithms.dpo.setup>`
  - ```{autodoc2-docstring} algorithms.dpo.setup
    :summary:
    ```
* - {py:obj}`add_ref_logprobs_to_data <algorithms.dpo.add_ref_logprobs_to_data>`
  - ```{autodoc2-docstring} algorithms.dpo.add_ref_logprobs_to_data
    :summary:
    ```
* - {py:obj}`validate <algorithms.dpo.validate>`
  - ```{autodoc2-docstring} algorithms.dpo.validate
    :summary:
    ```
* - {py:obj}`dpo_train <algorithms.dpo.dpo_train>`
  - ```{autodoc2-docstring} algorithms.dpo.dpo_train
    :summary:
    ```
````

### API

`````{py:class} DPOSaveState()
:canonical: algorithms.dpo.DPOSaveState

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} epoch
:canonical: algorithms.dpo.DPOSaveState.epoch
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOSaveState.epoch
```

````

````{py:attribute} step
:canonical: algorithms.dpo.DPOSaveState.step
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOSaveState.step
```

````

````{py:attribute} total_steps
:canonical: algorithms.dpo.DPOSaveState.total_steps
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOSaveState.total_steps
```

````

````{py:attribute} val_loss
:canonical: algorithms.dpo.DPOSaveState.val_loss
:type: float
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOSaveState.val_loss
```

````

````{py:attribute} consumed_samples
:canonical: algorithms.dpo.DPOSaveState.consumed_samples
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOSaveState.consumed_samples
```

````

`````

````{py:function} _default_dpo_save_state() -> algorithms.dpo.DPOSaveState
:canonical: algorithms.dpo._default_dpo_save_state

```{autodoc2-docstring} algorithms.dpo._default_dpo_save_state
```
````

`````{py:class} DPOConfig()
:canonical: algorithms.dpo.DPOConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} max_num_epochs
:canonical: algorithms.dpo.DPOConfig.max_num_epochs
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.max_num_epochs
```

````

````{py:attribute} max_num_steps
:canonical: algorithms.dpo.DPOConfig.max_num_steps
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.max_num_steps
```

````

````{py:attribute} val_period
:canonical: algorithms.dpo.DPOConfig.val_period
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.val_period
```

````

````{py:attribute} val_batches
:canonical: algorithms.dpo.DPOConfig.val_batches
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.val_batches
```

````

````{py:attribute} val_global_batch_size
:canonical: algorithms.dpo.DPOConfig.val_global_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.val_global_batch_size
```

````

````{py:attribute} val_micro_batch_size
:canonical: algorithms.dpo.DPOConfig.val_micro_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.val_micro_batch_size
```

````

````{py:attribute} val_at_start
:canonical: algorithms.dpo.DPOConfig.val_at_start
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.val_at_start
```

````

````{py:attribute} seed
:canonical: algorithms.dpo.DPOConfig.seed
:type: int
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.seed
```

````

````{py:attribute} reference_policy_kl_penalty
:canonical: algorithms.dpo.DPOConfig.reference_policy_kl_penalty
:type: float
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.reference_policy_kl_penalty
```

````

````{py:attribute} preference_average_log_probs
:canonical: algorithms.dpo.DPOConfig.preference_average_log_probs
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.preference_average_log_probs
```

````

````{py:attribute} sft_average_log_probs
:canonical: algorithms.dpo.DPOConfig.sft_average_log_probs
:type: bool
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.sft_average_log_probs
```

````

````{py:attribute} preference_loss_weight
:canonical: algorithms.dpo.DPOConfig.preference_loss_weight
:type: float
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.preference_loss_weight
```

````

````{py:attribute} sft_loss_weight
:canonical: algorithms.dpo.DPOConfig.sft_loss_weight
:type: float
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.DPOConfig.sft_loss_weight
```

````

`````

`````{py:class} MasterConfig()
:canonical: algorithms.dpo.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} policy
:canonical: algorithms.dpo.MasterConfig.policy
:type: nemo_rl.models.policy.PolicyConfig
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.MasterConfig.policy
```

````

````{py:attribute} data
:canonical: algorithms.dpo.MasterConfig.data
:type: nemo_rl.data.DataConfig
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.MasterConfig.data
```

````

````{py:attribute} dpo
:canonical: algorithms.dpo.MasterConfig.dpo
:type: algorithms.dpo.DPOConfig
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.MasterConfig.dpo
```

````

````{py:attribute} logger
:canonical: algorithms.dpo.MasterConfig.logger
:type: nemo_rl.utils.logger.LoggerConfig
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.MasterConfig.logger
```

````

````{py:attribute} cluster
:canonical: algorithms.dpo.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.MasterConfig.cluster
```

````

````{py:attribute} checkpointing
:canonical: algorithms.dpo.MasterConfig.checkpointing
:type: nemo_rl.utils.checkpoint.CheckpointingConfig
:value: >
   None

```{autodoc2-docstring} algorithms.dpo.MasterConfig.checkpointing
```

````

`````

````{py:function} setup(master_config: algorithms.dpo.MasterConfig, tokenizer: transformers.AutoTokenizer, train_dataset: nemo_rl.data.datasets.AllTaskProcessedDataset, val_dataset: nemo_rl.data.datasets.AllTaskProcessedDataset) -> tuple[nemo_rl.models.policy.lm_policy.Policy, nemo_rl.distributed.virtual_cluster.RayVirtualCluster, torchdata.stateful_dataloader.StatefulDataLoader, torchdata.stateful_dataloader.StatefulDataLoader, nemo_rl.algorithms.loss_functions.DPOLossFn, algorithms.dpo.MasterConfig, nemo_rl.utils.logger.Logger, nemo_rl.data.interfaces.TaskDataSpec, algorithms.dpo.DPOSaveState]
:canonical: algorithms.dpo.setup

```{autodoc2-docstring} algorithms.dpo.setup
```
````

````{py:function} add_ref_logprobs_to_data(dataloader, policy, master_config, is_val=False)
:canonical: algorithms.dpo.add_ref_logprobs_to_data

```{autodoc2-docstring} algorithms.dpo.add_ref_logprobs_to_data
```
````

````{py:function} validate(policy: nemo_rl.models.policy.interfaces.PolicyInterface, val_dataloader: torchdata.stateful_dataloader.StatefulDataLoader, tokenizer, loss_fn, step: int, master_config: algorithms.dpo.MasterConfig, val_batches: int, val_batch_size: int, val_mbs: int)
:canonical: algorithms.dpo.validate

```{autodoc2-docstring} algorithms.dpo.validate
```
````

````{py:function} dpo_train(policy, train_dataloader, val_dataloader, tokenizer, loss_fn, master_config, logger, checkpointer, dpo_save_state)
:canonical: algorithms.dpo.dpo_train

```{autodoc2-docstring} algorithms.dpo.dpo_train
```
````
