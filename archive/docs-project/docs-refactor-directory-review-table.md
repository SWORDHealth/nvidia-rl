# NeMo RL: Directory Structure Review

## Executive Summary

This document presents a **proposed** NeMo RL documentation directory structure that reorganizes the current flat structure (25 files) into a comprehensive hierarchical system (164 files) with clear progression from basic concepts to advanced topics.

**Please review the overall structure and provide comments in the review table below.**

**Key Metrics:**
- **Total Files**: 164 (77 main + 87 API docs)
- **Main Sections**: 8 organized categories
- **User Personas**: 4+ (beginners to researchers)
- **Learning Paths**: 3 structured progression paths

## Directory Structure Review Table

| Tree Structure | Description | Review Comments |
|----------------|-------------|-----------------|
| **docs/** | Main documentation landing page, build config, and static assets | |
| ├── index.md | Main documentation landing page | |
| ├── README.md | Documentation overview and structure | |
| ├── BUILD_INSTRUCTIONS.md | Build process and requirements | |
| ├── conf.py | Sphinx configuration | |
| ├── project.json | Project configuration | |
| ├── versions1.json | Version configuration | |
| ├── test_json_output.py | JSON output testing | |
| ├── assets/ | Static assets | |
| ├── _static/ | Static files | |
| ├── _extensions/ | Custom Sphinx extensions | |
| ├── _build/ | Build output | |
| | | |
| **about/** | Core project overview and introduction (3 files) | |
| ├── index.md | Main about page with project introduction | |
| ├── key-features.md | NeMo RL key features and capabilities | |
| └── architecture-overview.md | High-level system architecture | |
| | | |
| **get-started/** | User onboarding and setup (7 files) | |
| ├── index.md | Getting started landing page with learning paths | |
| ├── installation.md | Step-by-step installation guide | |
| ├── quickstart.md | Quick start tutorial for first-time users | |
| ├── docker.md | Containerized deployment guide | |
| ├── cluster.md | Multi-node cluster configuration | |
| ├── local-workstation.md | Local development environment setup | |
| └── model-selection.md | Guide for choosing appropriate models | |
| | | |
| **learning-resources/** | Educational content (13 files) | |
| ├── index.md | Learning resources landing page | |
| ├── tutorials/ | Step-by-step tutorials (4 files) | |
| │ ├── index.md | Tutorials overview and navigation | |
| │ ├── custom-environments.md | Custom environment development tutorial | |
| │ ├── custom-loss-functions.md | Custom loss function development tutorial | |
| │ └── distributed-training-scaling.md | Distributed training tutorial | |
| ├── examples/ | Working code examples (3 files) | |
| │ ├── index.md | Examples overview and navigation | |
| │ ├── sft-openmathinstruct2.md | SFT training on OpenMathInstruct dataset | |
| │ └── grpo-deepscaler.md | GRPO training on DeepScaler model | |
| └── use-cases/ | Real-world applications (5 files) | |
| ├── index.md | Use cases overview and navigation | |
| ├── mathematical-reasoning.md | Mathematical reasoning RLHF application | |
| ├── code-generation.md | Code generation with RLHF training | |
| ├── conversational-ai.md | Conversational AI applications | |
| └── scientific-research.md | Scientific research applications | |
| | | |
| **guides/** | Practical implementation guides (15 files) | |
| ├── index.md | Main guides page with navigation | |
| ├── troubleshooting.md | Comprehensive troubleshooting guide | |
| ├── training-algorithms/ | Algorithm-specific guides (5 files) | |
| │ ├── index.md | Training algorithms overview | |
| │ ├── sft.md | Supervised Fine-Tuning implementation | |
| │ ├── dpo.md | Direct Preference Optimization guide | |
| │ ├── grpo.md | Group Relative Policy Optimization | |
| │ └── eval.md | Model evaluation metrics and assessment | |
| ├── model-development/ | Model development workflows (3 files) | |
| │ ├── index.md | Model development overview | |
| │ ├── adding-new-models.md | Guide for integrating custom models | |
| │ └── model-quirks.md | Known model-specific behaviors | |
| ├── environment-data/ | Data and environment setup (4 files) | |
| │ ├── index.md | Environment and data management overview | |
| │ ├── environment-development.md | Custom environment development guide | |
| │ ├── debugging.md | Environment debugging and troubleshooting | |
| │ └── nsys-profiling.md | Performance profiling with NSight Systems | |
| └── training-optimization/ | Training optimization (3 files) | |
| ├── index.md | Training optimization overview | |
| ├── hyperparameter-optimization.md | Hyperparameter optimization guide | |
| ├── learning-rate-scheduling.md | Learning rate scheduling strategies | |
| └── training-stability.md | Training stability and convergence | |
| | | |
| **advanced/** | Research and performance optimization (20 files) | |
| ├── index.md | Advanced topics landing page | |
| ├── performance/ | Performance optimization (6 files) | |
| │ ├── index.md | Performance optimization overview | |
| │ ├── distributed-training.md | Multi-GPU and multi-node training | |
| │ ├── profiling.md | Performance profiling and analysis | |
| │ ├── monitoring.md | Real-time performance monitoring | |
| │ ├── memory-optimization.md | Memory usage optimization | |
| │ └── benchmarking.md | Performance benchmarking | |
| ├── research/ | Research methodologies (7 files) | |
| │ ├── index.md | Research methodologies overview | |
| │ ├── reproducible-research-validation.md | Reproducible research practices | |
| │ ├── performance-analysis.md | Performance analysis methodologies | |
| │ ├── experimental-design-validation.md | Experimental design and methodology | |
| │ ├── ablation-studies.md | Ablation study design and analysis | |
| │ ├── custom-algorithms.md | Custom algorithm development | |
| │ └── model-evaluation-validation.md | Model evaluation and validation | |
| └── algorithm-development/ | Algorithm development (5 files) | |
| ├── index.md | Algorithm development overview | |
| ├── custom-dpo.md | Custom DPO implementation | |
| ├── hyperparameter-optimization.md | Hyperparameter optimization | |
| ├── loss-functions.md | Loss function development | |
| └── mathematical-foundations.md | Mathematical foundations | |
| | | |
| **core-design/** | Architecture and design documents (16 files) | |
| ├── index.md | System design documentation landing page | |
| ├── design-principles/ | Core system design (4 files) | |
| │ ├── index.md | Core system architecture and components | |
| │ ├── design-and-philosophy.md | System design principles | |
| │ ├── generation.md | Text generation architecture | |
| │ └── fsdp2-parallel-plan.md | FSDP2 distributed training architecture | |
| ├── computational-systems/ | Computational design (3 files) | |
| │ ├── index.md | Computational infrastructure overview | |
| │ ├── training-backends.md | Training backend systems | |
| │ └── logger.md | Logging and monitoring infrastructure | |
| ├── data-management/ | Data architecture (4 files) | |
| │ ├── index.md | Data processing and management systems | |
| │ ├── padding.md | Data padding strategies | |
| │ ├── chat-datasets.md | Chat dataset processing | |
| │ └── checkpointing.md | Data checkpointing and recovery | |
| └── development-infrastructure/ | Dev infrastructure (4 files) | |
| ├── index.md | Development tools and infrastructure | |
| ├── loss-functions.md | Loss function implementations | |
| ├── checkpointing.md | Model checkpointing and recovery | |
| └── uv.md | UV package management system | |
| | | |
| **api-docs/** | Technical reference and API docs (87 files) | |
| ├── index.md | Complete API documentation overview | |
| ├── auto-generated.md | Auto-generation information | |
| ├── index.rst | RST API documentation structure | |
| ├── models.md | Model API reference | |
| ├── distributed.md | Distributed computing API reference | |
| ├── converters.md | Model converters API documentation | |
| └── nemo_rl/ | Complete NeMo RL API reference | |
| | | |
| **references/** | Tools and reference materials (3 files) | |
| ├── index.md | References overview and navigation | |
| ├── configuration-reference.md | Configuration file format and options | |
| └── cli-reference.md | Complete CLI command reference | |
