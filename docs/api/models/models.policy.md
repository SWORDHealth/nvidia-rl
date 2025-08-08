# {py:mod}`models.policy`

```{py:module} models.policy
```

```{autodoc2-docstring} models.policy
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

models.policy.dtensor_policy_worker
models.policy.fsdp1_policy_worker
models.policy.interfaces
models.policy.lm_policy
models.policy.megatron_policy_worker
models.policy.utils
```

## Package Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DTensorConfig <models.policy.DTensorConfig>`
  -
* - {py:obj}`SequencePackingConfig <models.policy.SequencePackingConfig>`
  -
* - {py:obj}`MegatronOptimizerConfig <models.policy.MegatronOptimizerConfig>`
  -
* - {py:obj}`MegatronSchedulerConfig <models.policy.MegatronSchedulerConfig>`
  -
* - {py:obj}`MegatronDDPConfig <models.policy.MegatronDDPConfig>`
  -
* - {py:obj}`MegatronConfig <models.policy.MegatronConfig>`
  -
* - {py:obj}`TokenizerConfig <models.policy.TokenizerConfig>`
  -
* - {py:obj}`PytorchOptimizerConfig <models.policy.PytorchOptimizerConfig>`
  -
* - {py:obj}`SinglePytorchSchedulerConfig <models.policy.SinglePytorchSchedulerConfig>`
  -
* - {py:obj}`DynamicBatchingConfig <models.policy.DynamicBatchingConfig>`
  -
* - {py:obj}`PolicyConfig <models.policy.PolicyConfig>`
  -
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SchedulerMilestones <models.policy.SchedulerMilestones>`
  - ```{autodoc2-docstring} models.policy.SchedulerMilestones
    :summary:
    ```
````

### API

`````{py:class} DTensorConfig()
:canonical: models.policy.DTensorConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: models.policy.DTensorConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.enabled
```

````

````{py:attribute} cpu_offload
:canonical: models.policy.DTensorConfig.cpu_offload
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.cpu_offload
```

````

````{py:attribute} sequence_parallel
:canonical: models.policy.DTensorConfig.sequence_parallel
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.sequence_parallel
```

````

````{py:attribute} activation_checkpointing
:canonical: models.policy.DTensorConfig.activation_checkpointing
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.activation_checkpointing
```

````

````{py:attribute} tensor_parallel_size
:canonical: models.policy.DTensorConfig.tensor_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.tensor_parallel_size
```

````

````{py:attribute} context_parallel_size
:canonical: models.policy.DTensorConfig.context_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.context_parallel_size
```

````

````{py:attribute} custom_parallel_plan
:canonical: models.policy.DTensorConfig.custom_parallel_plan
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.DTensorConfig.custom_parallel_plan
```

````

`````

`````{py:class} SequencePackingConfig()
:canonical: models.policy.SequencePackingConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: models.policy.SequencePackingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.SequencePackingConfig.enabled
```

````

````{py:attribute} train_mb_tokens
:canonical: models.policy.SequencePackingConfig.train_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.SequencePackingConfig.train_mb_tokens
```

````

````{py:attribute} logprob_mb_tokens
:canonical: models.policy.SequencePackingConfig.logprob_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.SequencePackingConfig.logprob_mb_tokens
```

````

````{py:attribute} algorithm
:canonical: models.policy.SequencePackingConfig.algorithm
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.SequencePackingConfig.algorithm
```

````

`````

`````{py:class} MegatronOptimizerConfig()
:canonical: models.policy.MegatronOptimizerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} optimizer
:canonical: models.policy.MegatronOptimizerConfig.optimizer
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.optimizer
```

````

````{py:attribute} lr
:canonical: models.policy.MegatronOptimizerConfig.lr
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.lr
```

````

````{py:attribute} min_lr
:canonical: models.policy.MegatronOptimizerConfig.min_lr
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.min_lr
```

````

````{py:attribute} weight_decay
:canonical: models.policy.MegatronOptimizerConfig.weight_decay
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.weight_decay
```

````

````{py:attribute} bf16
:canonical: models.policy.MegatronOptimizerConfig.bf16
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.bf16
```

````

````{py:attribute} fp16
:canonical: models.policy.MegatronOptimizerConfig.fp16
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.fp16
```

````

````{py:attribute} params_dtype
:canonical: models.policy.MegatronOptimizerConfig.params_dtype
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.params_dtype
```

````

````{py:attribute} adam_beta1
:canonical: models.policy.MegatronOptimizerConfig.adam_beta1
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.adam_beta1
```

````

````{py:attribute} adam_beta2
:canonical: models.policy.MegatronOptimizerConfig.adam_beta2
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.adam_beta2
```

````

````{py:attribute} adam_eps
:canonical: models.policy.MegatronOptimizerConfig.adam_eps
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.adam_eps
```

````

````{py:attribute} sgd_momentum
:canonical: models.policy.MegatronOptimizerConfig.sgd_momentum
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.sgd_momentum
```

````

````{py:attribute} use_distributed_optimizer
:canonical: models.policy.MegatronOptimizerConfig.use_distributed_optimizer
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.use_distributed_optimizer
```

````

````{py:attribute} use_precision_aware_optimizer
:canonical: models.policy.MegatronOptimizerConfig.use_precision_aware_optimizer
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.use_precision_aware_optimizer
```

````

````{py:attribute} clip_grad
:canonical: models.policy.MegatronOptimizerConfig.clip_grad
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronOptimizerConfig.clip_grad
```

````

`````

`````{py:class} MegatronSchedulerConfig()
:canonical: models.policy.MegatronSchedulerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} start_weight_decay
:canonical: models.policy.MegatronSchedulerConfig.start_weight_decay
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.start_weight_decay
```

````

````{py:attribute} end_weight_decay
:canonical: models.policy.MegatronSchedulerConfig.end_weight_decay
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.end_weight_decay
```

````

````{py:attribute} weight_decay_incr_style
:canonical: models.policy.MegatronSchedulerConfig.weight_decay_incr_style
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.weight_decay_incr_style
```

````

````{py:attribute} lr_decay_style
:canonical: models.policy.MegatronSchedulerConfig.lr_decay_style
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.lr_decay_style
```

````

````{py:attribute} lr_decay_iters
:canonical: models.policy.MegatronSchedulerConfig.lr_decay_iters
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.lr_decay_iters
```

````

````{py:attribute} lr_warmup_iters
:canonical: models.policy.MegatronSchedulerConfig.lr_warmup_iters
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.lr_warmup_iters
```

````

````{py:attribute} lr_warmup_init
:canonical: models.policy.MegatronSchedulerConfig.lr_warmup_init
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronSchedulerConfig.lr_warmup_init
```

````

`````

`````{py:class} MegatronDDPConfig()
:canonical: models.policy.MegatronDDPConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} grad_reduce_in_fp32
:canonical: models.policy.MegatronDDPConfig.grad_reduce_in_fp32
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronDDPConfig.grad_reduce_in_fp32
```

````

````{py:attribute} overlap_grad_reduce
:canonical: models.policy.MegatronDDPConfig.overlap_grad_reduce
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronDDPConfig.overlap_grad_reduce
```

````

````{py:attribute} overlap_param_gather
:canonical: models.policy.MegatronDDPConfig.overlap_param_gather
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronDDPConfig.overlap_param_gather
```

````

````{py:attribute} average_in_collective
:canonical: models.policy.MegatronDDPConfig.average_in_collective
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronDDPConfig.average_in_collective
```

````

````{py:attribute} use_custom_fsdp
:canonical: models.policy.MegatronDDPConfig.use_custom_fsdp
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronDDPConfig.use_custom_fsdp
```

````

````{py:attribute} data_parallel_sharding_strategy
:canonical: models.policy.MegatronDDPConfig.data_parallel_sharding_strategy
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronDDPConfig.data_parallel_sharding_strategy
```

````

`````

`````{py:class} MegatronConfig()
:canonical: models.policy.MegatronConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: models.policy.MegatronConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.enabled
```

````

````{py:attribute} empty_unused_memory_level
:canonical: models.policy.MegatronConfig.empty_unused_memory_level
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.empty_unused_memory_level
```

````

````{py:attribute} activation_checkpointing
:canonical: models.policy.MegatronConfig.activation_checkpointing
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.activation_checkpointing
```

````

````{py:attribute} converter_type
:canonical: models.policy.MegatronConfig.converter_type
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.converter_type
```

````

````{py:attribute} tensor_model_parallel_size
:canonical: models.policy.MegatronConfig.tensor_model_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.tensor_model_parallel_size
```

````

````{py:attribute} pipeline_model_parallel_size
:canonical: models.policy.MegatronConfig.pipeline_model_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.pipeline_model_parallel_size
```

````

````{py:attribute} num_layers_in_first_pipeline_stage
:canonical: models.policy.MegatronConfig.num_layers_in_first_pipeline_stage
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.num_layers_in_first_pipeline_stage
```

````

````{py:attribute} num_layers_in_last_pipeline_stage
:canonical: models.policy.MegatronConfig.num_layers_in_last_pipeline_stage
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.num_layers_in_last_pipeline_stage
```

````

````{py:attribute} context_parallel_size
:canonical: models.policy.MegatronConfig.context_parallel_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.context_parallel_size
```

````

````{py:attribute} pipeline_dtype
:canonical: models.policy.MegatronConfig.pipeline_dtype
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.pipeline_dtype
```

````

````{py:attribute} sequence_parallel
:canonical: models.policy.MegatronConfig.sequence_parallel
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.sequence_parallel
```

````

````{py:attribute} optimizer
:canonical: models.policy.MegatronConfig.optimizer
:type: typing.NotRequired[models.policy.MegatronOptimizerConfig]
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.optimizer
```

````

````{py:attribute} scheduler
:canonical: models.policy.MegatronConfig.scheduler
:type: typing.NotRequired[models.policy.MegatronSchedulerConfig]
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.scheduler
```

````

````{py:attribute} distributed_data_parallel_config
:canonical: models.policy.MegatronConfig.distributed_data_parallel_config
:type: models.policy.MegatronDDPConfig
:value: >
   None

```{autodoc2-docstring} models.policy.MegatronConfig.distributed_data_parallel_config
```

````

`````

`````{py:class} TokenizerConfig()
:canonical: models.policy.TokenizerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} name
:canonical: models.policy.TokenizerConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.TokenizerConfig.name
```

````

````{py:attribute} chat_template
:canonical: models.policy.TokenizerConfig.chat_template
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.TokenizerConfig.chat_template
```

````

`````

`````{py:class} PytorchOptimizerConfig()
:canonical: models.policy.PytorchOptimizerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} name
:canonical: models.policy.PytorchOptimizerConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.PytorchOptimizerConfig.name
```

````

````{py:attribute} kwargs
:canonical: models.policy.PytorchOptimizerConfig.kwargs
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} models.policy.PytorchOptimizerConfig.kwargs
```

````

`````

`````{py:class} SinglePytorchSchedulerConfig()
:canonical: models.policy.SinglePytorchSchedulerConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} name
:canonical: models.policy.SinglePytorchSchedulerConfig.name
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.SinglePytorchSchedulerConfig.name
```

````

````{py:attribute} kwargs
:canonical: models.policy.SinglePytorchSchedulerConfig.kwargs
:type: dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} models.policy.SinglePytorchSchedulerConfig.kwargs
```

````

`````

````{py:data} SchedulerMilestones
:canonical: models.policy.SchedulerMilestones
:value: >
   None

```{autodoc2-docstring} models.policy.SchedulerMilestones
```

````

`````{py:class} DynamicBatchingConfig()
:canonical: models.policy.DynamicBatchingConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} enabled
:canonical: models.policy.DynamicBatchingConfig.enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.DynamicBatchingConfig.enabled
```

````

````{py:attribute} train_mb_tokens
:canonical: models.policy.DynamicBatchingConfig.train_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.DynamicBatchingConfig.train_mb_tokens
```

````

````{py:attribute} logprob_mb_tokens
:canonical: models.policy.DynamicBatchingConfig.logprob_mb_tokens
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.DynamicBatchingConfig.logprob_mb_tokens
```

````

````{py:attribute} sequence_length_round
:canonical: models.policy.DynamicBatchingConfig.sequence_length_round
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.DynamicBatchingConfig.sequence_length_round
```

````

`````

`````{py:class} PolicyConfig()
:canonical: models.policy.PolicyConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} model_name
:canonical: models.policy.PolicyConfig.model_name
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.model_name
```

````

````{py:attribute} tokenizer
:canonical: models.policy.PolicyConfig.tokenizer
:type: models.policy.TokenizerConfig
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.tokenizer
```

````

````{py:attribute} train_global_batch_size
:canonical: models.policy.PolicyConfig.train_global_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.train_global_batch_size
```

````

````{py:attribute} train_micro_batch_size
:canonical: models.policy.PolicyConfig.train_micro_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.train_micro_batch_size
```

````

````{py:attribute} learning_rate
:canonical: models.policy.PolicyConfig.learning_rate
:type: float
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.learning_rate
```

````

````{py:attribute} logprob_batch_size
:canonical: models.policy.PolicyConfig.logprob_batch_size
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.logprob_batch_size
```

````

````{py:attribute} generation
:canonical: models.policy.PolicyConfig.generation
:type: typing.Optional[nemo_rl.models.generation.interfaces.GenerationConfig]
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.generation
```

````

````{py:attribute} generation_batch_size
:canonical: models.policy.PolicyConfig.generation_batch_size
:type: typing.NotRequired[int]
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.generation_batch_size
```

````

````{py:attribute} precision
:canonical: models.policy.PolicyConfig.precision
:type: str
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.precision
```

````

````{py:attribute} dtensor_cfg
:canonical: models.policy.PolicyConfig.dtensor_cfg
:type: models.policy.DTensorConfig
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.dtensor_cfg
```

````

````{py:attribute} megatron_cfg
:canonical: models.policy.PolicyConfig.megatron_cfg
:type: models.policy.MegatronConfig
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.megatron_cfg
```

````

````{py:attribute} dynamic_batching
:canonical: models.policy.PolicyConfig.dynamic_batching
:type: models.policy.DynamicBatchingConfig
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.dynamic_batching
```

````

````{py:attribute} sequence_packing
:canonical: models.policy.PolicyConfig.sequence_packing
:type: models.policy.SequencePackingConfig
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.sequence_packing
```

````

````{py:attribute} make_sequence_length_divisible_by
:canonical: models.policy.PolicyConfig.make_sequence_length_divisible_by
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.make_sequence_length_divisible_by
```

````

````{py:attribute} max_total_sequence_length
:canonical: models.policy.PolicyConfig.max_total_sequence_length
:type: int
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.max_total_sequence_length
```

````

````{py:attribute} max_grad_norm
:canonical: models.policy.PolicyConfig.max_grad_norm
:type: typing.Optional[typing.Union[float, int]]
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.max_grad_norm
```

````

````{py:attribute} fsdp_offload_enabled
:canonical: models.policy.PolicyConfig.fsdp_offload_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.fsdp_offload_enabled
```

````

````{py:attribute} activation_checkpointing_enabled
:canonical: models.policy.PolicyConfig.activation_checkpointing_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.activation_checkpointing_enabled
```

````

````{py:attribute} optimizer
:canonical: models.policy.PolicyConfig.optimizer
:type: typing.NotRequired[models.policy.PytorchOptimizerConfig]
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.optimizer
```

````

````{py:attribute} scheduler
:canonical: models.policy.PolicyConfig.scheduler
:type: typing.NotRequired[list[models.policy.SinglePytorchSchedulerConfig] | models.policy.SchedulerMilestones]
:value: >
   None

```{autodoc2-docstring} models.policy.PolicyConfig.scheduler
```

````

`````
