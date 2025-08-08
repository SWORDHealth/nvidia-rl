API Reference
=============

NeMo RL's API reference provides comprehensive technical documentation for all modules, classes, and functions. Use these references to understand the technical foundation of NeMo RL and integrate it with your reinforcement learning workflows.

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: :octicon:`brain;1.5em;sd-mr-1` Core Package
      :link: api/nemo_rl/nemo_rl
      :link-type: doc
      :class-card: sd-border-0

      **Main Interface**

      Primary API for NeMo RL package initialization, configuration, and core utilities.

      :bdg-secondary:`initialization` :bdg-secondary:`configuration` :bdg-secondary:`utilities`

   .. grid-item-card:: :octicon:`gear;1.5em;sd-mr-1` Algorithms
      :link: api/algorithms/algorithms
      :link-type: doc
      :class-card: sd-border-0

      **RL Algorithms**

      Implementation of reinforcement learning algorithms including DPO, GRPO, SFT, and custom loss functions.

      :bdg-secondary:`DPO` :bdg-secondary:`GRPO` :bdg-secondary:`SFT` :bdg-secondary:`loss functions`

   .. grid-item-card:: :octicon:`database;1.5em;sd-mr-1` Data Management
      :link: api/data/data
      :link-type: doc
      :class-card: sd-border-0

      **Data Handling**

      Data processing, dataset management, and HuggingFace integration for RL training workflows.

      :bdg-secondary:`datasets` :bdg-secondary:`huggingface` :bdg-secondary:`processing` :bdg-secondary:`interfaces`

   .. grid-item-card:: :octicon:`globe;1.5em;sd-mr-1` Environments
      :link: api/environments/environments
      :link-type: doc
      :class-card: sd-border-0

      **RL Environments**

      Reinforcement learning environments including math environments, games, and custom environment interfaces.

      :bdg-secondary:`environments` :bdg-secondary:`games` :bdg-secondary:`math` :bdg-secondary:`interfaces`

   .. grid-item-card:: :octicon:`graph;1.5em;sd-mr-1` Distributed Computing
      :link: api/distributed/distributed
      :link-type: doc
      :class-card: sd-border-0

      **Distributed Training**

      Distributed computing utilities for multi-GPU and multi-node RL training with Ray integration.

      :bdg-secondary:`distributed` :bdg-secondary:`ray` :bdg-secondary:`multi-gpu` :bdg-secondary:`collectives`

   .. grid-item-card:: :octicon:`play;1.5em;sd-mr-1` Experience Management
      :link: api/experience/experience
      :link-type: doc
      :class-card: sd-border-0

      **Experience Collection**

      Experience replay, rollout management, and trajectory collection for RL training.

      :bdg-secondary:`rollouts` :bdg-secondary:`experience` :bdg-secondary:`trajectories` :bdg-secondary:`replay`

   .. grid-item-card:: :octicon:`chart-line;1.5em;sd-mr-1` Evaluation
      :link: api/evals/evals
      :link-type: doc
      :class-card: sd-border-0

      **Model Evaluation**

      Evaluation frameworks and metrics for assessing RL model performance and behavior.

      :bdg-secondary:`evaluation` :bdg-secondary:`metrics` :bdg-secondary:`assessment` :bdg-secondary:`performance`

   .. grid-item-card:: :octicon:`tools;1.5em;sd-mr-1` Utilities
      :link: api/utils/utils
      :link-type: doc
      :class-card: sd-border-0

      **Utility Functions**

      Helper utilities, configuration management, and common tools for RL development.

      :bdg-secondary:`utilities` :bdg-secondary:`config` :bdg-secondary:`helpers` :bdg-secondary:`tools`

   .. grid-item-card:: :octicon:`arrow-left-right;1.5em;sd-mr-1` Converters
      :link: api/converters/converters
      :link-type: doc
      :class-card: sd-border-0

      **Model Converters**

      Model format conversion utilities for HuggingFace, Megatron, and other model formats.

      :bdg-secondary:`huggingface` :bdg-secondary:`megatron` :bdg-secondary:`conversion` :bdg-secondary:`formats`

   .. grid-item-card:: :octicon:`bar-chart;1.5em;sd-mr-1` Metrics
      :link: api/metrics/metrics
      :link-type: doc
      :class-card: sd-border-0

      **Performance Metrics**

      Training metrics, evaluation scores, and performance monitoring utilities.

      :bdg-secondary:`metrics` :bdg-secondary:`monitoring` :bdg-secondary:`scores` :bdg-secondary:`performance`

   .. grid-item-card:: :octicon:`cpu;1.5em;sd-mr-1` Models
      :link: api/models/models
      :link-type: doc
      :class-card: sd-border-0

      **Model Definitions**

      Neural network architectures, model components, and custom model implementations.

      :bdg-secondary:`models` :bdg-secondary:`architectures` :bdg-secondary:`networks` :bdg-secondary:`components`

.. toctree::
   :maxdepth: 1
   :caption: API Modules
   :hidden:

   api/nemo_rl/nemo_rl
   api/algorithms/algorithms
   api/data/data
   api/environments/environments
   api/distributed/distributed
   api/experience/experience
   api/evals/evals
   api/utils/utils
   api/converters/converters
   api/metrics/metrics
   api/models/models
