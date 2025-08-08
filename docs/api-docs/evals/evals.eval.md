# {py:mod}`evals.eval`

```{py:module} evals.eval
```

```{autodoc2-docstring} evals.eval
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EvalConfig <evals.eval.EvalConfig>`
  -
* - {py:obj}`MasterConfig <evals.eval.MasterConfig>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`setup <evals.eval.setup>`
  - ```{autodoc2-docstring} evals.eval.setup
    :summary:
    ```
* - {py:obj}`eval_pass_k <evals.eval.eval_pass_k>`
  - ```{autodoc2-docstring} evals.eval.eval_pass_k
    :summary:
    ```
* - {py:obj}`run_env_eval <evals.eval.run_env_eval>`
  - ```{autodoc2-docstring} evals.eval.run_env_eval
    :summary:
    ```
````

### API

`````{py:class} EvalConfig()
:canonical: evals.eval.EvalConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} metric
:canonical: evals.eval.EvalConfig.metric
:type: str
:value: >
   None

```{autodoc2-docstring} evals.eval.EvalConfig.metric
```

````

````{py:attribute} num_tests_per_prompt
:canonical: evals.eval.EvalConfig.num_tests_per_prompt
:type: int
:value: >
   None

```{autodoc2-docstring} evals.eval.EvalConfig.num_tests_per_prompt
```

````

````{py:attribute} seed
:canonical: evals.eval.EvalConfig.seed
:type: int
:value: >
   None

```{autodoc2-docstring} evals.eval.EvalConfig.seed
```

````

````{py:attribute} pass_k_value
:canonical: evals.eval.EvalConfig.pass_k_value
:type: int
:value: >
   None

```{autodoc2-docstring} evals.eval.EvalConfig.pass_k_value
```

````

`````

`````{py:class} MasterConfig()
:canonical: evals.eval.MasterConfig

Bases: {py:obj}`typing.TypedDict`

````{py:attribute} eval
:canonical: evals.eval.MasterConfig.eval
:type: evals.eval.EvalConfig
:value: >
   None

```{autodoc2-docstring} evals.eval.MasterConfig.eval
```

````

````{py:attribute} generate
:canonical: evals.eval.MasterConfig.generate
:type: nemo_rl.models.generation.interfaces.GenerationConfig
:value: >
   None

```{autodoc2-docstring} evals.eval.MasterConfig.generate
```

````

````{py:attribute} data
:canonical: evals.eval.MasterConfig.data
:type: nemo_rl.data.MathDataConfig
:value: >
   None

```{autodoc2-docstring} evals.eval.MasterConfig.data
```

````

````{py:attribute} env
:canonical: evals.eval.MasterConfig.env
:type: nemo_rl.environments.math_environment.MathEnvConfig
:value: >
   None

```{autodoc2-docstring} evals.eval.MasterConfig.env
```

````

````{py:attribute} cluster
:canonical: evals.eval.MasterConfig.cluster
:type: nemo_rl.distributed.virtual_cluster.ClusterConfig
:value: >
   None

```{autodoc2-docstring} evals.eval.MasterConfig.cluster
```

````

`````

````{py:function} setup(master_config: evals.eval.MasterConfig, tokenizer: transformers.AutoTokenizer, dataset: nemo_rl.data.datasets.AllTaskProcessedDataset) -> tuple[nemo_rl.models.generation.vllm.VllmGeneration, torch.utils.data.DataLoader, evals.eval.MasterConfig]
:canonical: evals.eval.setup

```{autodoc2-docstring} evals.eval.setup
```
````

````{py:function} eval_pass_k(rewards: torch.Tensor, num_tests_per_prompt: int, k: int) -> float
:canonical: evals.eval.eval_pass_k

```{autodoc2-docstring} evals.eval.eval_pass_k
```
````

````{py:function} run_env_eval(vllm_generation, dataloader, env, master_config)
:canonical: evals.eval.run_env_eval

```{autodoc2-docstring} evals.eval.run_env_eval
```
````
