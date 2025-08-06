# NeMo RL: Refactor PR Plan

This document organizes all files in the `/docs` directory by logical groupings for efficient GitHub PR assignments and developer review. There are approximately **80 .md files** that need to be reviewed across the 6 PR groups (excluding 81 auto-generated API documentation files).

## Table of Contents

- [PR Group 1: Core Setup & User Onboarding](#pr-group-1-core-setup--user-onboarding)
- [PR Group 2: Training Algorithms](#pr-group-2-training-algorithms)
- [PR Group 3: Tutorials & Learning Resources](#pr-group-3-tutorials--learning-resources)
- [PR Group 4: Examples & Use Cases](#pr-group-4-examples--use-cases)
- [PR Group 5: Model Development & Environment](#pr-group-5-model-development--environment)
- [PR Group 6: Core Architecture & Design](#pr-group-6-core-architecture--design)
- [PR Group 7: Advanced Performance & Research](#pr-group-7-advanced-performance--research)
- [PR Group 8: API Documentation & References](#pr-group-8-api-documentation--references)
- [Total Count Summary](#total-count-summary)
- [Documentation Review Outline](#documentation-review-outline)
  - [Review Criteria](#review-criteria)
  - [Review Process Steps](#review-process-steps)

---

## PR Group 1: Core Setup & User Onboarding
**Focus:** Essential installation guides, environment setup, first-time user onboarding, and project overview
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/index.md` | Main documentation landing page and navigation hub |
| `docs/get-started/index.md` | Get started landing page with setup overview |
| `docs/get-started/installation.md` | Step-by-step installation guide for NeMo RL |
| `docs/get-started/quickstart.md` | Quick start tutorial for first-time users |
| `docs/get-started/local-workstation.md` | Local development environment setup guide |
| `docs/get-started/cluster.md` | Multi-node cluster configuration and setup |
| `docs/get-started/docker.md` | Containerized deployment with Docker |
| `docs/get-started/model-selection.md` | Guide for choosing appropriate models |
| `docs/about/index.md` | Project overview and introduction |
| `docs/about/key-features.md` | Key features and capabilities |
| `docs/about/architecture-overview.md` | High-level system architecture |

---

## PR Group 2: Training Algorithms
**Focus:** Core RLHF training algorithms and evaluation methods
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/guides/training-algorithms/index.md` | Overview of all supported RLHF training algorithms |
| `docs/guides/training-algorithms/sft.md` | Supervised Fine-Tuning implementation and configuration |
| `docs/guides/training-algorithms/dpo.md` | Direct Preference Optimization algorithm guide |
| `docs/guides/training-algorithms/grpo.md` | Group Relative Policy Optimization implementation |
| `docs/guides/training-algorithms/eval.md` | Model evaluation metrics and assessment methods |

---

## PR Group 3: Tutorials & Learning Resources
**Focus:** Step-by-step tutorials and educational content
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/learning-resources/index.md` | Learning resources landing page and navigation |
| `docs/learning-resources/tutorials/index.md` | Complete tutorials overview and navigation |
| `docs/learning-resources/tutorials/custom-environments.md` | Custom environment development tutorial |
| `docs/learning-resources/tutorials/custom-loss-functions.md` | Custom loss function development tutorial |
| `docs/learning-resources/tutorials/distributed-training-scaling.md` | Distributed training tutorial |

---

## PR Group 4: Examples & Use Cases
**Focus:** Real-world examples and practical applications
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/learning-resources/examples/index.md` | Real-world examples and case studies |
| `docs/learning-resources/examples/sft-openmathinstruct2.md` | SFT training on OpenMathInstruct dataset |
| `docs/learning-resources/examples/grpo-deepscaler.md` | GRPO training on DeepScaler model |
| `docs/learning-resources/use-cases/index.md` | Practical use cases and applications |
| `docs/learning-resources/use-cases/mathematical-reasoning.md` | Mathematical reasoning RLHF application |
| `docs/learning-resources/use-cases/code-generation.md` | Code generation with RLHF training |
| `docs/learning-resources/use-cases/conversational-ai.md` | Conversational AI applications |
| `docs/learning-resources/use-cases/scientific-research.md` | Scientific research applications |

---

## PR Group 5: Model Development & Environment
**Focus:** Custom model integration, environment development, debugging tools, performance profiling, and training optimization
**Priority:** High

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/guides/model-development/index.md` | Model development and integration overview |
| `docs/guides/model-development/adding-new-models.md` | Guide for integrating custom models into NeMo RL |
| `docs/guides/model-development/model-quirks.md` | Known model-specific behaviors and workarounds |
| `docs/guides/environment-data/index.md` | Environment and data management overview |
| `docs/guides/environment-data/environment-development.md` | Custom environment development guide |
| `docs/guides/environment-data/debugging.md` | Environment debugging and troubleshooting |
| `docs/guides/environment-data/nsys-profiling.md` | Performance profiling with NVIDIA NSight Systems |
| `docs/guides/training-optimization/index.md` | Training optimization overview |
| `docs/guides/training-optimization/hyperparameter-optimization.md` | Hyperparameter optimization guide |
| `docs/guides/training-optimization/learning-rate-scheduling.md` | Learning rate scheduling strategies |
| `docs/guides/training-optimization/training-stability.md` | Training stability and convergence |
| `docs/guides/troubleshooting.md` | Comprehensive troubleshooting guide |

---

## PR Group 6: Core Architecture & Design
**Focus:** System design, core architecture, and development infrastructure
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/core-design/index.md` | System design documentation landing page |
| `docs/core-design/design-principles/index.md` | Core system architecture and components |
| `docs/core-design/design-principles/design-and-philosophy.md` | System design principles and philosophy |
| `docs/core-design/design-principles/generation.md` | Text generation architecture and flow |
| `docs/core-design/design-principles/fsdp2-parallel-plan.md` | FSDP2 distributed training architecture |
| `docs/core-design/data-management/index.md` | Data processing and management systems |
| `docs/core-design/data-management/padding.md` | Data padding strategies and implementation |
| `docs/core-design/data-management/chat-datasets.md` | Chat dataset processing and formatting |
| `docs/core-design/data-management/checkpointing.md` | Data checkpointing and recovery |
| `docs/core-design/computational-systems/index.md` | Computational infrastructure overview |
| `docs/core-design/computational-systems/training-backends.md` | Training backend systems and engines |
| `docs/core-design/computational-systems/logger.md` | Logging and monitoring infrastructure |
| `docs/core-design/development-infrastructure/index.md` | Development tools and infrastructure |
| `docs/core-design/development-infrastructure/loss-functions.md` | Loss function implementations and design |
| `docs/core-design/development-infrastructure/checkpointing.md` | Model checkpointing and recovery |
| `docs/core-design/development-infrastructure/uv.md` | UV package management system |

---

## PR Group 7: Advanced Performance & Research
**Focus:** Performance optimization, research methodologies, and algorithm development
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/advanced/index.md` | Advanced topics and optimization landing page |
| `docs/advanced/performance/index.md` | Performance optimization overview |
| `docs/advanced/performance/distributed-training.md` | Multi-GPU and multi-node training |
| `docs/advanced/performance/profiling.md` | Performance profiling and analysis tools |
| `docs/advanced/performance/monitoring.md` | Real-time performance monitoring |
| `docs/advanced/performance/memory-optimization.md` | Memory usage optimization strategies |
| `docs/advanced/performance/benchmarking.md` | Performance benchmarking and comparison |
| `docs/advanced/research/index.md` | Research methodologies and practices overview |
| `docs/advanced/research/reproducible-research-validation.md` | Reproducible research practices and standards |
| `docs/advanced/research/performance-analysis.md` | Performance analysis methodologies |
| `docs/advanced/research/experimental-design-validation.md` | Experimental design and methodology |
| `docs/advanced/research/ablation-studies.md` | Ablation study design and analysis |
| `docs/advanced/research/custom-algorithms.md` | Custom algorithm development and testing |
| `docs/advanced/research/model-evaluation-validation.md` | Model evaluation and validation |
| `docs/advanced/algorithm-development/index.md` | Algorithm development overview |
| `docs/advanced/algorithm-development/custom-dpo.md` | Custom DPO implementation |
| `docs/advanced/algorithm-development/hyperparameter-optimization.md` | Hyperparameter optimization |
| `docs/advanced/algorithm-development/loss-functions.md` | Loss function development |
| `docs/advanced/algorithm-development/mathematical-foundations.md` | Mathematical foundations |

---

## PR Group 8: API Documentation & References
**Focus:** Complete API reference, distributed computing interfaces, CLI tools, and configuration management
**Priority:** Medium

### Files:

| File Path | Description |
|-----------|-------------|
| `docs/api-docs/index.md` | Complete API documentation overview |
| `docs/api-docs/auto-generated.md` | Auto-generated API documentation |
| `docs/api-docs/converters.md` | Model converters API |
| `docs/api-docs/distributed.md` | Distributed computing API reference |
| `docs/api-docs/models.md` | Models API interfaces and classes |
| `docs/api-docs/nemo_rl/` | Complete NeMo RL API reference (81 files) |
| `docs/references/index.md` | References overview and navigation |
| `docs/references/cli-reference.md` | Complete CLI command reference |
| `docs/references/configuration-reference.md` | Configuration file format and options |

---

## Total Count Summary

### Total Count: 82 files
Breakdown:
- **get-started/**: 7 files
- **guides/**: 15 files
- **core-design/**: 16 files
- **learning-resources/**: 13 files
- **advanced/**: 20 files
- **api-docs/**: 6 main files + 81 auto-generated API files
- **references/**: 3 files
- **about/**: 3 files

---

## Documentation Review Outline

### **Review Process Overview**
This outline provides a structured approach for reviewing the NeMo RL documentation transformation across all 6 PR groups.

### **Review Criteria**

#### **1. Content Quality Assessment**
- **Accuracy**: Technical content is correct and up-to-date
- **Completeness**: All necessary information is included
- **Migration Coverage**: All source content from archive has been successfully migrated to new documentation structure
- **Clarity**: Content is clear and understandable
- **Consistency**: Formatting and style are consistent across files
- **Relevance**: Content is relevant to the target audience

#### **2. Structure and Organization**
- **Logical flow**: Information is organized in a logical sequence
- **Navigation**: Users can easily find what they need
- **Cross-references**: Links between related content work properly
- **Hierarchy**: Information is properly categorized and nested

#### **3. User Experience**
- **Accessibility**: Content is accessible to all users
- **Readability**: Text is easy to read and scan
- **Visual design**: Cards, badges, and layout enhance usability
- **Mobile responsiveness**: Content works on different devices

#### **4. Technical Implementation**
- **Markdown formatting**: Proper MyST-Parser syntax
- **Frontmatter**: Complete and accurate metadata
- **Links**: All internal and external links work
- **Build system**: Content builds without errors

### **Review Process Steps**

#### **Step 1: Pre-Review Preparation**
1. **Review the PR group structure** and understand file relationships
2. **Set up local build environment** to test documentation
3. **Review the review criteria** and checklist for the specific PR group
4. **Identify key stakeholders** who should review specific content

#### **Step 2: Content Review**
1. **Read through each file** systematically
2. **Test all links and cross-references**
3. **Verify technical accuracy** against source code
4. **Check formatting and visual consistency**
5. **Assess user experience and navigation**

#### **Step 3: Feedback and Iteration**
1. **Document issues** with specific file paths and line numbers
2. **Prioritize feedback** by impact and severity
3. **Provide constructive suggestions** for improvements
4. **Coordinate with other reviewers** to avoid duplicate feedback
5. **Follow up on critical issues** until resolved

#### **Step 4: Final Approval**
1. **Verify all critical issues are resolved**
2. **Confirm build system works correctly**
3. **Test navigation and user experience**
4. **Approve PR when all criteria are met**
5. **Merge PR into main branch** after approval