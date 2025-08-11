---
description: "Comprehensive API documentation for NeMo RL framework including distributed computing, models, algorithms, and data interfaces"
categories: ["reference"]
tags: ["api", "reference", "distributed", "models", "algorithms", "data-interfaces", "python-api"]
personas: ["mle-focused", "researcher-focused"]
difficulty: "reference"
content_type: "reference"
modality: "universal"
---

# About NeMo RL API Documentation

Welcome to the NeMo RL API documentation! This section provides comprehensive reference documentation for all the APIs, interfaces, and components that make up the NeMo RL framework.

## Overview

NeMo RL provides a modular and extensible API for reinforcement learning with large language models. The framework is built around several core abstractions that enable scalable, distributed training and inference across multiple backends.

## Core Architecture

NeMo RL is designed around four key capabilities that every RL system needs:

1. **Resource Management**: Allocate and manage compute resources (GPUs/CPUs)
2. **Isolation**: Provide isolated process environments for different components
3. **Coordination**: Control and orchestrate distributed components
4. **Communication**: Enable data flow between components

These capabilities are implemented through a set of composable abstractions that scale from single GPU to thousands of GPUs.

## Quick Navigation

::::{grid} 1 2 2 2
:gutter: 2 2 2 2

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Distributed Computing
:link: nemo_rl/nemo_rl.distributed
:link-type: doc

Core distributed computing abstractions including VirtualCluster and WorkerGroup.

+++
{bdg-primary}`Core`
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Models and Policies
:link: nemo_rl/nemo_rl.models
:link-type: doc

Model interfaces, policy implementations, and generation backends.

+++
{bdg-info}`Models`
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Algorithms
:link: nemo_rl/nemo_rl.algorithms
:link-type: doc

RL algorithms including DPO, GRPO, SFT, and custom loss functions.

+++
{bdg-warning}`Training`
:::

:::{grid-item-card} {octicon}`database;1.5em;sd-mr-1` Data and Environments
:link: nemo_rl/nemo_rl.data
:link-type: doc

Data processing, dataset interfaces, and environment implementations.

+++
{bdg-secondary}`Data`
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Converters
:link: converters
:link-type: doc

Model conversion and export utilities for deployment and production.

+++
{bdg-info}`Deployment`
:::

::::

## Key Components

### Distributed Computing

The distributed computing layer provides abstractions for managing compute resources and coordinating distributed processes:

- **VirtualCluster**: Manages resource allocation and placement groups
- **WorkerGroup**: Coordinates groups of distributed worker processes
- **BatchedDataDict**: Efficient data structures for distributed communication

### Models and Policies

The model layer provides interfaces for different model backends and policy implementations:

- **PolicyInterface**: Abstract interface for RL policies
- **GenerationInterface**: Unified interface for text generation backends
- **Model Backends**: Support for Hugging Face, Megatron, and custom backends

### Algorithms

The algorithms layer implements various RL algorithms and training methods:

- **DPO**: Direct Preference Optimization
- **GRPO**: Group Relative Policy Optimization  
- **SFT**: Supervised Fine-Tuning
- **Custom Loss Functions**: Extensible loss function framework

### Data and Environments

The data layer handles data processing, dataset management, and environment interactions:

- **Dataset Interfaces**: Standardized dataset loading and processing
- **Environment Interfaces**: RL environment abstractions
- **Data Processing**: Tokenization, batching, and preprocessing utilities

### Converters

The converters layer provides model conversion and export utilities:

- **HuggingFace/Megatron Converters**: vLLM export and Megatron → HuggingFace
- **Deployment Utilities**: Model export and production deployment tools

## Design Philosophy

NeMo RL follows several key design principles:

### Modular Abstractions

Each component is designed as a modular abstraction that can be composed and extended:

```python
# Example: Composing a generation pipeline
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.vllm import VllmGeneration, VllmConfig

cluster = RayVirtualCluster(bundle_ct_per_node_list=[4], use_gpus=True, num_gpus_per_node=4)
vllm_cfg: VllmConfig = {
    "backend": "vllm",
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "max_new_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": None,
    "pad_token_id": 0,
    "stop_token_ids": [],
    "vllm_cfg": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "gpu_memory_utilization": 0.8, "max_model_len": 4096, "skip_tokenizer_init": True, "async_engine": False, "precision": "bfloat16", "load_format": "auto"},
}
generator = VllmGeneration(cluster, vllm_cfg)
```

### Backend Independence

The framework is designed to be backend-agnostic, allowing easy switching between different implementations:

```python
# Same interface, different backends (via configuration in Policy)
# See nemo_rl.models.policy.lm_policy.Policy and example configs in examples/configs/
```

### Scalability

The abstractions scale seamlessly from single GPU to thousands of GPUs:

```python
# Single GPU
cluster = RayVirtualCluster([1])  # 1 GPU

# Multi-GPU
cluster = RayVirtualCluster([8, 8])  # 2 nodes, 8 GPUs each

# Same code works at any scale
worker_group = RayWorkerGroup(cluster, policy_class)
```

## Getting Started

### Basic Usage

```python
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.algorithms.dpo import setup as dpo_setup, dpo_train

# Set up distributed environment
cluster = RayVirtualCluster(bundle_ct_per_node_list=[4], use_gpus=True, num_gpus_per_node=4)

# See examples/run_dpo.py for a complete training pipeline using Policy and dpo_train
```

### Custom Components

Extending NeMo RL with custom components is straightforward:

```python
from nemo_rl.models.interfaces import PolicyInterface

class CustomPolicy(PolicyInterface):
    def generate(self, batch):
        # Custom generation logic
        pass
    
    def train(self, batch, loss_fn):
        # Custom training logic
        pass
```

## API Reference

The following sections provide detailed API documentation for each component:

- [Distributed Computing](nemo_rl/nemo_rl.distributed): VirtualCluster, WorkerGroup, and distributed utilities
- [Models and Policies](nemo_rl/nemo_rl.models): Policy interfaces, generation backends, and model implementations
- [Algorithms](nemo_rl/nemo_rl.algorithms): RL algorithms, loss functions, and training utilities
- [Data and Environments](nemo_rl/nemo_rl.data): Dataset interfaces, environment abstractions, and data processing
- [Converters](nemo_rl/nemo_rl.converters): Model conversion and export utilities for deployment
- [Utilities](nemo_rl/nemo_rl.utils): Logging, configuration, and utility functions
- [Auto-Generated Reference](auto-generated): Complete API reference with all functions, classes, and parameters
- [Complete API Reference](nemo_rl/nemo_rl): Full auto-generated documentation for all modules



