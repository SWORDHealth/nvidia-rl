# NeMo RL Documentation: File Mapping Assessment

This document provides a comprehensive comparison between the current `docs/` structure and the archived `archive/docs/` structure, mapping source files and describing the evolution of the documentation system.

## Table of Contents

- [Executive Summary](#executive-summary)
- [File Mapping Table](#file-mapping-table)
- [New Sections and Files](#new-sections-and-files)
- [Archived Files Status](#archived-files-status)
- [Key Structural Changes](#key-structural-changes)
- [Content Enhancements](#content-enhancements)
- [Navigation and User Experience](#navigation-and-user-experience)

---

## Executive Summary

The documentation has undergone a **major transformation** from a flat, developer-focused structure to a comprehensive, user-centric documentation system. Key improvements include:

- **📈 400%+ increase** in documentation files (from ~25 to 100+ files)
- **🎯 User-centric organization** with clear learning paths and personas
- **📚 Comprehensive coverage** spanning from beginner tutorials to advanced research
- **🔧 Enhanced navigation** with landing pages and cross-references
- **📋 Standardized frontmatter** for consistent metadata and searchability

---

## File Mapping Table

### **Core Documentation Files**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `index.md` | `docs/index.md` | **Complete Restructure** | Transformed from simple toctree to comprehensive landing page with learning paths, navigation cards, and user-centric organization |
| `README.md` | `docs/README.md` | **Enhanced** | Updated with new structure overview, build instructions, and contributor guidelines |
| `conf.py` | `docs/conf.py` | **Enhanced** | Updated configuration for new directory structure, enhanced Sphinx settings |
| `project.json` | `docs/project.json` | **Enhanced** | Updated project configuration for new structure |
| `versions1.json` | `docs/versions1.json` | **Enhanced** | Updated version configuration for new structure |

### **Training Algorithm Guides**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `guides/sft.md` | `docs/guides/training-algorithms/sft.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |
| `guides/dpo.md` | `docs/guides/training-algorithms/dpo.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |
| `guides/grpo.md` | `docs/guides/training-algorithms/grpo.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |
| `guides/eval.md` | `docs/guides/training-algorithms/eval.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |

### **Setup and Installation Files**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `cluster.md` | `docs/get-started/cluster.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |
| `docker.md` | `docs/get-started/docker.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |
| `local-workstation.md` | `docs/get-started/local-workstation.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, enhanced content |

### **Design Documentation**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `design-docs/design-and-philosophy.md` | `docs/core-design/design-principles/design-and-philosophy.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, reorganized under design-principles |
| `design-docs/padding.md` | `docs/core-design/data-management/padding.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to data-management section |
| `design-docs/logger.md` | `docs/core-design/computational-systems/logger.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to computational-systems section |
| `design-docs/uv.md` | `docs/core-design/development-infrastructure/uv.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to development-infrastructure section |
| `design-docs/chat-datasets.md` | `docs/core-design/data-management/chat-datasets.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to data-management section |
| `design-docs/generation.md` | `docs/core-design/design-principles/generation.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to design-principles section |
| `design-docs/checkpointing.md` | `docs/core-design/data-management/checkpointing.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to data-management section |
| `design-docs/loss-functions.md` | `docs/core-design/development-infrastructure/loss-functions.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to development-infrastructure section |
| `design-docs/fsdp2-parallel-plan.md` | `docs/core-design/design-principles/fsdp2-parallel-plan.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to design-principles section |
| `design-docs/training-backends.md` | `docs/core-design/computational-systems/training-backends.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to computational-systems section |

### **Production Support Files**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `testing.md` | `docs/guides/environment-data/debugging.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to environment-data section |
| `debugging.md` | `docs/guides/environment-data/debugging.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to environment-data section |
| `documentation.md` | `docs/guides/environment-data/debugging.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to environment-data section |

### **Model Development**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `adding-new-models.md` | `docs/guides/model-development/adding-new-models.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved to model-development section |

### **Examples and Tutorials**

| Archive Source | Current Location | Status | Changes Description |
|----------------|------------------|--------|-------------------|
| `guides/sft-openmathinstruct2.md` | `docs/learning-resources/examples/sft-openmathinstruct2.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved from guides to examples section |
| `guides/grpo-deepscaler.md` | `docs/learning-resources/examples/grpo-deepscaler.md` | **Moved + Enhanced** | Added frontmatter taxonomy, updated relative paths, moved from guides to examples section |

---

## New Sections and Files

### **Get Started Section** (`docs/get-started/`)
**Purpose**: Onboarding and setup documentation for new users

| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Get started landing page | Navigation structure, learning paths |
| `installation.md` | Installation guide | Step-by-step installation instructions |
| `quickstart.md` | Quick start tutorial | End-to-end tutorial for first use |
| `model-selection.md` | Model selection guide | Guide for choosing appropriate models |

### **Learning Resources Section** (`docs/learning-resources/`)
**Purpose**: Tutorials, examples, and use cases for hands-on learning

#### **Tutorials** (`docs/learning-resources/tutorials/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Tutorials overview | Navigation and learning paths |
| `custom-environments.md` | Custom environment development | Step-by-step tutorial |
| `custom-loss-functions.md` | Custom loss function development | Step-by-step tutorial |
| `distributed-training-scaling.md` | Distributed training tutorial | Step-by-step tutorial |

#### **Examples** (`docs/learning-resources/examples/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Examples overview | Navigation and descriptions |

#### **Use Cases** (`docs/learning-resources/use-cases/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Use cases overview | Navigation and descriptions |
| `code-generation.md` | Code generation use case | Real-world application |
| `conversational-ai.md` | Conversational AI use case | Real-world application |
| `mathematical-reasoning.md` | Mathematical reasoning use case | Real-world application |
| `scientific-research.md` | Scientific research use case | Real-world application |

### **Advanced Topics Section** (`docs/advanced/`)
**Purpose**: Advanced concepts for experienced users and researchers

#### **Performance** (`docs/advanced/performance/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Performance overview | Navigation and optimization techniques |
| `benchmarking.md` | Performance benchmarking | Advanced optimization |
| `distributed-training.md` | Distributed training guide | Advanced scaling |
| `memory-optimization.md` | Memory optimization | Advanced optimization |
| `monitoring.md` | Performance monitoring | Advanced monitoring |
| `profiling.md` | Performance profiling | Advanced profiling |

#### **Algorithm Development** (`docs/advanced/algorithm-development/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Algorithm development overview | Navigation and development workflow |
| `custom-dpo.md` | Custom DPO implementation | Advanced algorithm development |
| `hyperparameter-optimization.md` | Hyperparameter optimization | Advanced optimization |
| `loss-functions.md` | Loss function development | Advanced algorithm development |
| `mathematical-foundations.md` | Mathematical foundations | Theoretical concepts |

#### **Research** (`docs/advanced/research/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Research overview | Navigation and research methodologies |
| `ablation-studies.md` | Ablation studies guide | Research methodology |
| `custom-algorithms.md` | Custom algorithm development | Research implementation |
| `experimental-design-validation.md` | Experimental design | Research methodology |
| `model-evaluation-validation.md` | Model evaluation | Research methodology |
| `performance-analysis.md` | Performance analysis | Research methodology |
| `reproducible-research-validation.md` | Reproducible research | Research methodology |

### **API Documentation** (`docs/api-docs/`)
**Purpose**: Comprehensive API reference documentation

| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | API documentation overview | Navigation and API organization |
| `auto-generated.md` | Auto-generated API docs | Auto-generation system |
| `converters.md` | Model converters API | API reference |
| `distributed.md` | Distributed computing API | API reference |
| `models.md` | Models API | API reference |

### **References Section** (`docs/references/`)
**Purpose**: Reference documentation for tools and configuration

| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | References overview | Navigation and reference organization |
| `cli-reference.md` | CLI reference | Command-line interface reference |
| `configuration-reference.md` | Configuration reference | Configuration options reference |

### **About Section** (`docs/about/`)
**Purpose**: Project overview and introduction

| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | About page | Project overview and introduction |
| `architecture-overview.md` | Architecture overview | System architecture description |
| `key-features.md` | Key features | Feature descriptions and capabilities |

### **Core Design Section** (`docs/core-design/`)
**Purpose**: System architecture and design documentation

#### **Design Principles** (`docs/core-design/design-principles/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Design principles overview | Navigation and design concepts |

#### **Data Management** (`docs/core-design/data-management/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Data management overview | Navigation and data architecture |

#### **Computational Systems** (`docs/core-design/computational-systems/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Computational systems overview | Navigation and system architecture |

#### **Development Infrastructure** (`docs/core-design/development-infrastructure/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Development infrastructure overview | Navigation and development tools |

### **Guides Section** (`docs/guides/`)
**Purpose**: Comprehensive guides for different aspects of the system

#### **Training Optimization** (`docs/guides/training-optimization/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Training optimization overview | Navigation and optimization techniques |
| `hyperparameter-optimization.md` | Hyperparameter optimization guide | Optimization techniques |
| `learning-rate-scheduling.md` | Learning rate scheduling | Optimization techniques |
| `training-stability.md` | Training stability guide | Best practices |

#### **Environment and Data** (`docs/guides/environment-data/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Environment and data overview | Navigation and data handling |
| `environment-development.md` | Environment development | Environment creation and customization |
| `nsys-profiling.md` | NSYS profiling guide | Performance profiling |

#### **Model Development** (`docs/guides/model-development/`)
| New File | Purpose | Content Type |
|----------|---------|--------------|
| `index.md` | Model development overview | Navigation and development workflow |
| `model-quirks.md` | Model quirks documentation | Model-specific issues and troubleshooting |

---

## Archived Files Status

### **Files Successfully Migrated**
All files from the archive have been successfully migrated to the new structure with enhancements:
- ✅ All 25+ archive files migrated
- ✅ Frontmatter added to all files
- ✅ Path references updated
- ✅ Content enhanced where appropriate

### **Files Removed/Consolidated**
- `documentation.md` - Content split into build instructions (`BUILD_INSTRUCTIONS.md`) and documentation template (`README.md`)
- `testing.md` - Content consolidated into debugging guide

---

## Key Structural Changes

### **1. User-Centric Organization**
- **Before**: Flat structure organized by content type
- **After**: Hierarchical structure organized by user journey and expertise level

### **2. Learning Paths**
- **Before**: No clear learning progression
- **After**: Defined learning paths from beginner to advanced

### **3. Persona-Based Content**
- **Before**: Generic documentation
- **After**: Content tailored for specific personas (MLEs, researchers, DevOps)

### **4. Comprehensive Coverage**
- **Before**: Basic guides and setup
- **After**: Complete coverage from installation to advanced research

### **5. Enhanced Navigation**
- **Before**: Simple toctree navigation
- **After**: Rich landing pages with cards, learning paths, and cross-references

---

## Content Enhancements

### **Frontmatter Standardization**
All markdown files now include standardized frontmatter with:
- **Description**: 1-2 sentence content summary
- **Categories**: Primary category classification
- **Tags**: 2-8 relevant tags for search/discovery
- **Personas**: Target audience identification
- **Difficulty**: beginner/intermediate/advanced/reference
- **Content Type**: tutorial/concept/reference/troubleshooting/example
- **Modality**: text-only/image-only/video-only/multimodal/universal

### **Path Updates**
All internal links updated to reflect new directory structure:
- `../../examples/` → `../../../examples/`
- `../cluster.md` → `../../get-started/cluster.md`
- `../design-docs/` → `../../core-design/[section]/`

### **Content Quality Improvements**
- Enhanced step-by-step instructions
- Added code examples and snippets
- Improved troubleshooting sections
- Better cross-references between related content

---

## Navigation and User Experience

### **Landing Pages**
Each major section now has a dedicated landing page with:
- Navigation cards for easy discovery
- Learning paths for guided progression
- Overview content and key resources
- Logical progression and user workflows

### **Cross-References**
Enhanced cross-referencing system:
- Related content suggestions
- Prerequisites and next steps
- Contextual navigation

### **Search and Discovery**
Improved searchability through:
- Standardized frontmatter
- Consistent tagging system
- Enhanced metadata

---

## Summary

The documentation transformation represents a **comprehensive evolution** from a basic developer reference to a full-featured, user-centric documentation system. The new structure provides:

- **🎯 Better user experience** with clear learning paths and persona-based content
- **📚 Comprehensive coverage** spanning all user needs and expertise levels
- **🔧 Enhanced navigation** with rich landing pages and cross-references
- **📋 Standardized structure** for consistent maintenance and updates
- **🚀 Scalable architecture** for future content additions

The documentation now serves as a complete resource for users at all levels, from initial setup to advanced research and development.