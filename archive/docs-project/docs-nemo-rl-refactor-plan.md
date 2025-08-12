---
description: "Consolidated NeMo RL documentation refactor plan with mapping, PR groups, review process, and deployment timeline"
categories: ["internal"]
tags: ["refactor-plan", "documentation", "nemo-rl"]
---

# NeMo RL Documentation Refactor Plan

This document presents the refactor proposal, PR plan, and mapping diff for the NeMo RL documentation project. It describes the goals, scope, structure, migration mapping, review process, and deployment timeline.

---

## Summary

- Goal: Transform NeMo RL docs from a flat structure into a user-centric, hierarchical system with better navigation and coverage.
- Coverage increase: 25 → 82 manually authored files (+81 Auto‑Generated API docs).
 - Coverage increase: 25 → ~80 curated docs (+81 per‑module auto‑generated API docs).
- Organization: 8 major sections aligned to user journeys and personas.
- Status: Migration complete; review and deployment follow; optimization planned.

---

## Key Improvements

- ✅ Complete organizational restructuring with clear navigation
- ✅ Modern visual design with grid cards and landing pages
- ✅ Comprehensive coverage across use cases and personas
- ✅ Enhanced technical features (AI assistant, advanced search, JSON output, content gating)
- ✅ Professional standards: metadata, cross-references, consistent front matter
- ✅ Improved UX with learning paths and structured journeys

---

## Information Architecture

```text
docs/
├── about/                       # Project overview
├── get-started/                 # User onboarding
├── learning-resources/          # Tutorials, examples, use-cases
├── guides/                      # Practical guides
├── advanced/                    # Research & optimization
├── core-design/                 # Architecture docs
├── api-docs/                    # Technical reference (incl. auto-generated)
└── references/                  # CLI & configuration
```

## Final Documentation Structure (High‑Level)

| Section | Files | Purpose |
|---------|-------|---------|
| `about/` | 3 | Project overview |
| `get-started/` | 7 | Getting started |
| `learning-resources/` | 13 | Tutorials, examples, use cases |
| `guides/` | 18 | Practical how‑to guides |
| `core-design/` | 15 | Architecture and systems design |
| `references/` | 3 | CLI, configuration |
| `api-docs/` | Overview pages + 81 per‑module | API reference (overview + per‑module) |
| `_extensions/` | 8+ | Custom functionality |

---

## Personas And Learning Paths

- Personas: MLEs, Researchers, DevOps, Cluster Administrators
- Learning Paths:
  - Beginner: Installation → Quick Start → Model Selection → Examples
  - Intermediate: Configuration Reference → Distributed Training → Guides → Use Cases
  - Advanced: Troubleshooting → Performance → Core Design → References

## Content Standards And Enhancements

- Front Matter (desc, categories, tags, personas, difficulty, content type, modality); quality: step‑by‑step, cross‑references; navigation: landing pages and indices

---

## Migration and Mapping Summary

High‑level mapping from `archive/docs` to `docs` with enhanced content and reorganized hierarchy.

### Exact Mapping Table

| Source (archive) | Destination (docs) | Status | Description of changes |
|------------------|--------------------|--------|------------------------|
| `archive/docs/conf.py` | `docs/conf.py` | Moved | Sphinx configuration expanded; added custom extensions/templates |
| `archive/docs/index.md` | `docs/index.md` | Moved | Landing page expanded with cross‑links and cards |
| `archive/docs/versions1.json` | `docs/versions1.json` | Moved | Version configuration updated |
| `archive/docs/project.json` | `docs/project.json` | Moved | Project metadata updated |
| `archive/docs/docker.md` | `docs/get-started/docker.md` | Moved | Content enhanced and reorganized under Get Started |
| `archive/docs/local-workstation.md` | `docs/get-started/local-workstation.md` | Moved | Clarified local run guidance |
| `archive/docs/cluster.md` | `docs/get-started/cluster.md` | Moved | Expanded with Slurm workflow and tips |
| `archive/docs/adding-new-models.md` | `docs/guides/model-development/adding-new-models.md` | Moved | Guidance expanded and standardized |
| `archive/docs/documentation.md` | N/A | Removed | Superseded by the new structure |
| `archive/docs/testing.md` | `docs/guides/troubleshooting.md` | Moved | Material merged into troubleshooting/performance |
| `archive/docs/debugging.md` | `docs/guides/environment-data/debugging.md` | Moved | Updated and expanded |
| `archive/docs/guides/sft.md` | `docs/guides/training-algorithms/sft.md` | Moved | Enhanced algorithm guide |
| `archive/docs/guides/dpo.md` | `docs/guides/training-algorithms/dpo.md` | Moved | Enhanced algorithm guide |
| `archive/docs/guides/grpo.md` | `docs/guides/training-algorithms/grpo.md` | Moved | Enhanced algorithm guide |
| `archive/docs/guides/eval.md` | `docs/guides/training-algorithms/eval.md` | Moved | Evaluation guide standardized |
| `archive/docs/guides/sft-openmathinstruct2.md` | `docs/learning-resources/examples/sft-openmathinstruct2.md` | Moved | Reclassified under Examples |
| `archive/docs/guides/grpo-deepscaler.md` | `docs/learning-resources/examples/grpo-deepscaler.md` | Moved | Reclassified under Examples |
| `archive/docs/design-docs/design-and-philosophy.md` | `docs/core-design/design-principles/design-and-philosophy.md` | Moved | Content enhanced and retitled |
| `archive/docs/design-docs/generation.md` | `docs/core-design/design-principles/generation.md` | Moved | Content enhanced |
| `archive/docs/design-docs/fsdp2-parallel-plan.md` | `docs/core-design/design-principles/fsdp2-parallel-plan.md` | Moved | Content enhanced |
| `archive/docs/design-docs/loss-functions.md` | `docs/advanced/algorithm-development/loss-functions.md` | Moved | Content enhanced |
| `archive/docs/design-docs/training-backends.md` | `docs/core-design/computational-systems/training-backends.md` | Moved | Content enhanced |
| `archive/docs/design-docs/uv.md` | `docs/core-design/development-infrastructure/uv.md` | Moved | Content enhanced |
| `archive/docs/design-docs/logger.md` | `docs/core-design/computational-systems/logger.md` | Moved | Content enhanced |
| `archive/docs/design-docs/padding.md` | `docs/core-design/data-management/padding.md` | Moved | Content enhanced |
| `archive/docs/design-docs/checkpointing.md` | `docs/core-design/data-management/checkpointing.md` | Moved | Content enhanced |
| `archive/docs/design-docs/chat-datasets.md` | `docs/core-design/data-management/chat-datasets.md` | Moved | Content enhanced |
| `archive/docs/helpers.py` | N/A | Retained (archived) | Not part of active docs |

### Preserved and Enhanced

- Core configuration and landing content (`docs/conf.py`, `docs/index.md`) expanded and standardized
- Training algorithm guides (SFT, DPO, GRPO, Evaluation) enhanced and consistently structured
- Core Design content reorganized into design principles, data management, computational systems, and development infrastructure

### Removed (Content Integrated)

- `archive/docs/documentation.md` → replaced by the new hierarchical structure
- `archive/docs/autodoc2_docstrings_parser.py` → retained under `archive/`; not part of active docs

### New Content Areas

- Get Started: `installation.md`, `quickstart.md`, `docker.md`, `cluster.md`, `local-workstation.md`
- About: `about/index.md`, `architecture-overview.md`, `key-features.md`
- Learning Resources: `tutorials/`, `examples/`, `use-cases/`
- Core Design: design principles, computational systems, data management, development infrastructure
- References: `cli-reference.md`, `configuration-reference.md`, `index.md`
- API Docs: overview pages plus per‑module auto‑generated docs under `api-docs/nemo_rl/`

### Supporting Files

- `docs/README.md`—main docs README
- `docs/BUILD_INSTRUCTIONS.md`—build instructions
- `docs/versions1.json`—version configuration
- `docs/test_json_output.py`—JSON output test
- `docs/project.json`—project configuration

## Technical Infrastructure

- Sphinx configuration enhanced with extensions and templates
- Custom extensions: AI assistant, content gating, JSON output, enhanced search assets
- Support for better indexing and search relevance via JSON output and enhanced search assets
---

## Deployment Strategy

### Pull Request Groups (6)

1. Core Setup & Getting Started (6 Files)
   - Files for review:
     - `docs/conf.py`
     - `docs/index.md`
     - `docs/README.md`
     - `docs/get-started/index.md`
     - `docs/get-started/installation.md`
     - `docs/get-started/quickstart.md`

2. Configuration & Execution (4 Files)
   - Files for review:
     - `docs/references/configuration-reference.md`
     - `docs/references/cli-reference.md`
     - `docs/get-started/docker.md`
     - `docs/get-started/cluster.md`

3. Operations & Troubleshooting (4 Files)
   - Files for review:
     - `docs/guides/environment-data/index.md`
     - `docs/guides/troubleshooting.md`
     - `docs/references/index.md`
     - `docs/BUILD_INSTRUCTIONS.md`

4. Core Design & Advanced Performance (7 Files)
   - Files for review:
     - `docs/advanced/performance/index.md`
     - `docs/advanced/performance/distributed-training.md`
     - `docs/advanced/performance/profiling.md`
     - `docs/advanced/performance/monitoring.md`
     - `docs/core-design/computational-systems/training-backends.md`
     - `docs/core-design/development-infrastructure/uv.md`
     - `docs/core-design/computational-systems/logger.md`

5. API Documentation (7 Files)
   - Files for review:
     - `docs/api-docs/index.md`
     - `docs/api-docs/models.md`
     - `docs/api-docs/distributed.md`
     - `docs/api-docs/converters.md`
     - `docs/api-docs/auto-generated.md`
     - `docs/api-docs/nemo_rl/` (per‑module auto‑generated)
     - `docs/index.md` (API overview linkage)

6. About & Learning Resources (6 Files)
   - Files for review:
     - `docs/about/index.md`
     - `docs/about/architecture-overview.md`
     - `docs/about/key-features.md`
     - `docs/learning-resources/index.md`
     - `docs/learning-resources/tutorials/index.md`
     - `docs/learning-resources/examples/index.md`

---

## Total Count Summary

- ~80 curated docs + 81 per‑module auto‑generated API docs

Counts refer to curated, manually authored pages across non‑auto‑generated sections (including API overview pages), plus per‑module pages under `api-docs/nemo_rl/`.

---

## Implementation Status and Timeline

- Phase 1: Foundation—Complete
  - Structure design, content creation, extensions, review process setup
- Phase 2: Deployment—In Progress
  - Reviews, stakeholder feedback, QA, production deployment
- Phase 3: Optimization—Planned
  - User feedback, performance monitoring, search relevance tuning, content gap fixes, technical debt cleanup, maintainer enablement

---

## Risks And Mitigations

- Preserve archive in `archive/docs/`
- Gradual roll out across PR groups
- Maintain backward compatibility via link updates
- Quality assurance across content and technical layers
---

## Quality Assurance

- Review process: content, structure, UX, technical, integration
- Success indicators: improved navigation/search, modern platform capabilities, maintainability gains, expanded documentation coverage (82 curated + 81 auto‑generated)

---

## Conclusion

The refactor delivers a modern, scalable, and user‑centered documentation system with comprehensive coverage, clear learning paths, robust technical infrastructure, and a staged deployment plan that minimizes risk while maximizing usability and maintainability.
