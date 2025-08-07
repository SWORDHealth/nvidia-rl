# Documentation Structure Mapping: Archive vs Current

## Overview

This document maps the transition from the archive/docs structure to the current docs structure, showing file movements, reorganizations, and structural changes.

## File Mapping Table

| Archive Location | Current Location | Status | Notes |
|------------------|------------------|---------|-------|
| `archive/docs/index.md` | `docs/index.md` | ✅ Moved | Updated with new structure |
| `archive/docs/conf.py` | `docs/conf.py` | ✅ Moved | Enhanced configuration |
| `archive/docs/versions1.json` | `docs/versions1.json` | ✅ Moved | Updated content |
| `archive/docs/project.json` | `docs/project.json` | ✅ Moved | Updated content |
| `archive/docs/docker.md` | `docs/get-started/docker.md` | ✅ Moved | Enhanced content |
| `archive/docs/local-workstation.md` | `docs/get-started/local-workstation.md` | ✅ Moved | Enhanced content |
| `archive/docs/cluster.md` | `docs/get-started/cluster.md` | ✅ Moved | Enhanced content |
| `archive/docs/adding-new-models.md` | `docs/guides/model-development/adding-new-models.md` | ✅ Moved | Enhanced content |
| `archive/docs/documentation.md` | ❌ Removed | ❌ Deleted | Replaced by new structure |
| `archive/docs/testing.md` | `docs/guides/production-support/testing.md` | ✅ Moved | Enhanced content |
| `archive/docs/debugging.md` | `docs/guides/environment-data/debugging.md` | ✅ Moved | Enhanced content |
| `archive/docs/helpers.py` | ❌ Removed | ❌ Deleted | No longer needed |
| `archive/docs/autodoc2_docstrings_parser.py` | ❌ Removed | ❌ Deleted | No longer needed |

### Guides Directory Mapping

| Archive Location | Current Location | Status | Notes |
|------------------|------------------|---------|-------|
| `archive/docs/guides/grpo.md` | `docs/guides/training-algorithms/grpo.md` | ✅ Moved | Enhanced content |
| `archive/docs/guides/dpo.md` | `docs/guides/training-algorithms/dpo.md` | ✅ Moved | Enhanced content |
| `archive/docs/guides/sft.md` | `docs/guides/training-algorithms/sft.md` | ✅ Moved | Enhanced content |
| `archive/docs/guides/eval.md` | `docs/guides/training-algorithms/eval.md` | ✅ Moved | Enhanced content |
| `archive/docs/guides/sft-openmathinstruct2.md` | `docs/learning-resources/examples/sft-openmathinstruct2.md` | ✅ Moved | Moved to examples |
| `archive/docs/guides/grpo-deepscaler.md` | `docs/learning-resources/examples/grpo-deepscaler.md` | ✅ Moved | Moved to examples |

### Design Docs Directory Mapping

| Archive Location | Current Location | Status | Notes |
|------------------|------------------|---------|-------|
| `archive/docs/design-docs/design-and-philosophy.md` | `docs/core-design/design-principles/design-and-philosophy.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/generation.md` | `docs/core-design/design-principles/generation.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/fsdp2-parallel-plan.md` | `docs/core-design/design-principles/fsdp2-parallel-plan.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/loss-functions.md` | `docs/advanced/algorithm-development/loss-functions.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/training-backends.md` | `docs/core-design/computational-systems/training-backends.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/uv.md` | `docs/core-design/development-infrastructure/uv.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/logger.md` | `docs/core-design/computational-systems/logger.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/padding.md` | `docs/core-design/data-management/padding.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/checkpointing.md` | `docs/core-design/data-management/checkpointing.md` | ✅ Moved | Enhanced content |
| `archive/docs/design-docs/chat-datasets.md` | `docs/core-design/data-management/chat-datasets.md` | ✅ Moved | Enhanced content |

### Assets Directory Mapping

| Archive Location | Current Location | Status | Notes |
|------------------|------------------|---------|-------|
| `archive/docs/assets/*.png` | `docs/assets/*.png` | ✅ Moved | Some files may be updated/replaced |

## Structural Changes

### New Directory Structure

**Archive Structure (Flat):**
```
archive/docs/
├── guides/
├── design-docs/
├── assets/
└── [individual files]
```

**Current Structure (Hierarchical):**
```
docs/
├── get-started/
├── about/
├── core-design/
│   ├── design-principles/
│   ├── computational-systems/
│   ├── development-infrastructure/
│   └── data-management/
├── guides/
│   ├── training-algorithms/
│   ├── training-optimization/
│   ├── model-development/
│   └── environment-data/
├── advanced/
│   ├── performance/
│   ├── algorithm-development/
│   └── research/
├── learning-resources/
│   ├── tutorials/
│   ├── examples/
│   └── use-cases/
├── references/
├── api-docs/
│   └── nemo_rl/
├── assets/
├── _static/
└── _extensions/
```

## Key Changes Explained

### 1. **Reorganization by User Journey**
- **Before**: Flat structure with mixed content types
- **After**: Organized by user journey (get-started → guides → advanced → references)

### 2. **Enhanced Categorization**
- **Training algorithms** moved to dedicated subdirectory
- **Design docs** reorganized into core-design with subcategories
- **New sections** for learning-resources, advanced topics, and references

### 3. **Content Consolidation**
- **Examples moved**: Specific implementation examples moved to learning-resources/examples
- **Enhanced content**: All moved files show significant content expansion
- **Better organization**: Content organized by purpose rather than implementation

### 4. **New Infrastructure**
- Added `_static/` and `_extensions/` for enhanced functionality
- Added `api-docs/` with auto-generated documentation
- Added `references/` for CLI and configuration documentation

### 5. **Improved Navigation**
- Added index files in each directory for better navigation
- Created hierarchical structure for better content discovery
- Separated concerns (getting started vs advanced topics)

## Content Enhancements

### Files with Enhanced Content
- `docker.md`: Expanded from 2.1KB to 4.7KB
- `cluster.md`: Expanded from 6.9KB to 8.1KB
- `local-workstation.md`: Expanded from 1.6KB to 1.9KB
- `grpo.md`: Expanded from 13KB to 15KB
- `dpo.md`: Expanded from 7.6KB to 8.1KB

### New Content Areas
- **About section**: Architecture overview and key features
- **Learning resources**: Tutorials, examples, and use cases
- **Advanced topics**: Performance, algorithm development, research
- **References**: CLI and configuration documentation
- **API documentation**: Auto-generated and manual API docs

## Summary

The documentation has evolved from a flat, implementation-focused structure to a user-centric, hierarchical organization that better supports different user personas and use cases. The new structure provides clearer navigation, more comprehensive content, and better separation of concerns between getting started, advanced topics, and reference materials.

**Important Note**: All files from the archive were actually moved and enhanced rather than deleted. The specific examples (sft-openmathinstruct2, grpo-deepscaler) were moved to the learning-resources/examples section, and all design docs were reorganized into appropriate subdirectories within core-design and advanced sections.
