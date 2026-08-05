# iPSC Digital Twin Production Program

This repository is the production-oriented development layer for the iPSC two-stage perturbation digital twin platform. It is designed for a small scientific software team that needs reproducible experiment tracking, dataset versioning, treatment library management, model registration, validation, DOE recommendation, dashboard exports, and an optional LLM-assisted literature/CMC module.

This is not a clinical system. It does not make clinical claims, and it should be treated as a scientific decision-support and experimental planning platform.

## What This Program Covers

The Production program currently supports:
- Experiment registry for tracking multiple studies over time
- Dataset registry with dataset versions, hashes, schema version, and preprocessing version
- Treatment library loading, validation, and mapping into AnnData metadata
- Reproducible baseline model training with model artifacts, metrics, model cards, and model registry records
- Validation utilities with leakage-safe holdout strategies and condition-level metrics
- Active learning / DOE recommendation for next-round treatment selection
- Dashboard-ready exports for downstream apps or notebooks
- Optional literature and CMC evidence extraction/synthesis using an injected LLM client

## Repository Structure

```text
Production/
├── docs/
│   └── production_architecture.md
├── examples/
│   └── treatment_library.yaml
├── src/
│   └── ipsc_digital_twin/
│       ├── __init__.py
│       ├── active_learning.py
│       ├── cli.py
│       ├── dashboard_export.py
│       ├── model_registry.py
│       ├── training.py
│       ├── treatments.py
│       ├── literature/
│       │   ├── __init__.py
│       │   ├── extraction.py
│       │   ├── schemas.py
│       │   └── synthesis.py
│       ├── registry/
│       │   ├── __init__.py
│       │   ├── dataset_registry.py
│       │   └── experiment_registry.py
│       └── validation/
│           ├── __init__.py
│           ├── metrics.py
│           ├── reports.py
│           └── splits.py
├── tests/
├── pyproject.toml
└── README.md
```

## Module Guide

### `src/ipsc_digital_twin/registry/`
Registry layer for persistent experiment and dataset metadata.
- `experiment_registry.py`: create, validate, load, and list experiment records
- `dataset_registry.py`: register datasets, compute file hashes, assign dataset versions, and link them to experiments

### `src/ipsc_digital_twin/treatments.py`
Treatment library management.
- Load YAML treatment libraries
- Validate required treatment fields and enums
- Map `round1_treatment` / `round2_treatment` in AnnData to library records
- Report unknown treatments
- Generate treatment mechanism summaries

### `src/ipsc_digital_twin/training.py`
Reproducible training entrypoint for baseline and RegenAI-PT forward transition models.
- Fixed random-seed setup
- Model artifact save/load flow
- Metrics export
- Model card generation

### `src/ipsc_digital_twin/models/`
Forward modeling backends and neural training utilities.
- `regenai_pt_config.py`: configuration and AnnData validation for the RegenAI-PT backend
- `regenai_pt_data.py`: dataset, dataloader, and round-specific preparation utilities
- `regenai_pt_model.py`: encoder / perturbation / covariate / decoder / adversarial architecture
- `regenai_pt_losses.py`: reconstruction, adversarial, and regularization losses
- `regenai_pt_trainer.py`: PyTorch trainer with save/load and prediction helpers
- `regenai_pt_adapter.py`: adapter that exposes the neural forward model through the shared forward interface used by inverse recommendation
- `simulation.py`: round1, round2, and sequential A->B simulation helpers

### `src/ipsc_digital_twin/model_registry.py`
Model registry for tracking trained artifacts.
- Registers model type, dataset version, config, metrics, artifact hash, git commit, and random seed
- Supports listing and describing trained models

### `src/ipsc_digital_twin/validation/`
Production validation framework.
- Leakage-safe split strategies
- Ranking and regression-style metrics
- Markdown / JSON / CSV report generation

### `src/ipsc_digital_twin/active_learning.py`
Next-experiment DOE recommendation.
- Balances exploitation, exploration, diversity, and controls
- Produces DOE table, replicate assignments, plate map suggestions, and a Markdown rationale report

### `src/ipsc_digital_twin/dashboard_export.py`
Dashboard-ready artifact export.
- Writes JSON/CSV/Markdown outputs for frontend or reporting use
- Includes schema version, experiment ID, and dataset version in JSON payloads

### `src/ipsc_digital_twin/literature/`
Optional LLM-assisted literature and CMC support.
- Per-paper extraction with structured JSON
- Map-reduce synthesis across papers or protocol summaries
- Output generation for evidence and rationale tables
- Designed so tests can use a mock client with no external API calls

## Installation

### Dedicated Production Environment

Production now has its own virtual environment and should not rely on the MVP environment.
Use the Production-local interpreter:
- `/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production/.venv/bin/python`

### Development Install

```bash
cd "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production"
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .[dev]
```

Notes:
- On this machine, the macOS system `python3` was Python `3.9.6`, which is too old for the Production package requirement `>=3.10`.
- The dedicated Production environment was rebuilt and verified with Python `3.11.14` on July 30, 2026.
- If you recreate the environment later, prefer a Python `3.11+` interpreter.
- A standard local install is used on this machine because macOS marks `.pth` files inside the hidden `.venv` directory as hidden, which can prevent editable installs from adding `src/` to `sys.path`.
- For editable-style development, run commands with `PYTHONPATH=src`, for example `PYTHONPATH=src python -m pytest -q`.

### RegenAI-PT Dependencies

As of July 30, 2026, `torch 2.13.0` is installed and verified in the dedicated Production `.venv`, so the neural `regenai_pt` backend is available.

If you recreate the environment later, reinstall `torch` with:

```bash
cd "/Users/qiaoqiaowan/Desktop/iPSC/AI_iPSC_differentiation/Perterbation_predictor/Production"
source .venv/bin/activate
python -m pip install torch
```

Verify the scientific environment:

```bash
python -c "import torch, anndata, h5py; print(torch.__version__)"
```

### Verify The CLI

```bash
ipsc-twin --help
```

If the console script is not available yet, you can run:

```bash
python -m ipsc_digital_twin.cli --help
```

## RegenAI-PT Forward Transition Model

The Production program now includes a **RegenAI-PT forward transition model** that acts as the neural simulator behind inverse treatment recommendation.

### Purpose

The forward model is designed to learn:
- current expression + treatment + dose + covariates -> predicted future expression/state

This means the primary prediction target is **future cell state** rather than a single downstream `final_score`.
Condition-level scores can still be derived later, but they are downstream summaries rather than the core model target.

### Architecture

The RegenAI-PT backend uses the following components:
- **Encoder**: expression vector -> basal latent state `z_basal`
- **Component embeddings**: one learned embedding for each treatment component
- **Dose-response function**: component-specific dose scaling before aggregation
- **Interaction module**: optional learned pairwise interaction effect for multi-component treatments
- **Covariate embeddings**: additive embeddings for `iPSC_line`, `batch`, `round`, `time_point`, `replicate`, `sequencing_run`, and other configured covariates
- **Decoder**: perturbed latent state -> predicted future expression
- **Adversarial classifiers**: treatment and covariate probes attached to `z_basal` through gradient reversal
- **Combined loss**: reconstruction loss + adversarial losses + embedding regularization + dose regularization

This is **RegenAI-PT** in spirit and interface, but it is not presented as the official CPA implementation.

### Support for Round 1 and Round 2

The same backend supports the two-stage experiment structure:
- **Round 1**: `intermediate + round1_treatment -> post_round1`
- **Round 2**: `post_round1 + round2_treatment -> post_round2`
- **Sequential simulation**: `A -> B`

In round-2 mode, `round1_treatment` can also be included as a context covariate so the model can condition on treatment history.

### Multi-Component Treatments

Each round can contain one or more treatment components. The canonical AnnData fields are:

- `round1_components`, `round1_doses`, `round1_dose_units`
- `round2_components`, `round2_doses`, `round2_dose_units`

Each cell stores aligned lists, for example:

```python
adata.obs.at[cell_id, "round1_components"] = ["CHIR99021", "ActivinA"]
adata.obs.at[cell_id, "round1_doses"] = [3.0, 100.0]
adata.obs.at[cell_id, "round1_dose_units"] = ["uM", "ng/mL"]
```

Legacy `round1_treatment` / `round1_dose` and `round2_treatment` / `round2_dose` columns remain supported. A legacy treatment label is converted internally to a one-component treatment. Component effects are dose-scaled and summed; when enabled, the interaction MLP adds a learned pairwise effect.

Sequential simulations therefore support forms such as:

- Round 1: `["CHIR99021", "ActivinA"]`
- Round 2: `["RA", "FGF2"]`
- Sequence: `CHIR99021+ActivinA -> RA+FGF2`

### Relationship to the Inverse Recommender

The RegenAI-PT model is the **forward simulator**.
The inverse recommender does not directly guess the best treatment from scratch. Instead, it:
1. generates candidate one-step or two-step treatment sequences
2. simulates each candidate with the forward model
3. compares predicted future states to the requested target state
4. ranks treatments by target-state closeness, safety penalties, and any available uncertainty terms

Conceptually:
- **Forward model**: current state + treatment -> predicted future state
- **Inverse recommender**: search over candidate treatments using the forward model as the simulator

### CLI Examples

#### Train a RegenAI-PT model

```bash
ipsc-twin train \
  --registry-dir state \
  --dataset-id <DATASET_ID> \
  --model-type regenai_pt \
  --output-dir outputs/regenai_pt_training \
  --treatment-key round1_treatment \
  --dose-key round1_dose \
  --covariate-keys iPSC_line,batch,round,time_point \
  --max-epochs 50 \
  --batch-size 256 \
  --n-latent 64
```

#### Train round 1 explicitly

```bash
ipsc-twin train \
  --registry-dir state \
  --dataset-id <DATASET_ID> \
  --model-type regenai_pt \
  --output-dir outputs/regenai_pt_round1 \
  --treatment-key round1_treatment \
  --dose-key round1_dose \
  --covariate-keys iPSC_line,batch,round,time_point \
  --max-epochs 50 \
  --batch-size 256 \
  --n-latent 64
```

#### Train round 2 explicitly

```bash
ipsc-twin train \
  --registry-dir state \
  --dataset-id <DATASET_ID> \
  --model-type regenai_pt \
  --output-dir outputs/regenai_pt_round2 \
  --treatment-key round2_treatment \
  --dose-key round2_dose \
  --covariate-keys iPSC_line,batch,round,time_point,round1_treatment \
  --max-epochs 50
```

#### Train a combined round1+round2 simulator bundle

```bash
ipsc-twin train \
  --registry-dir state \
  --dataset-id <DATASET_ID> \
  --model-type regenai_pt \
  --treatment-mode combined_rounds \
  --output-dir outputs/regenai_pt_combined \
  --covariate-keys iPSC_line,batch,round,time_point \
  --max-epochs 50
```

#### Simulate a two-step sequence

```bash
ipsc-twin simulate-sequence \
  --model-path outputs/regenai_pt_combined/<MODEL_ARTIFACT>.pt \
  --adata data/current_cells.h5ad \
  --round1-treatment "Treatment_A" \
  --round1-dose 10 \
  --round2-treatment "Treatment_B" \
  --round2-dose 5 \
  --output-dir outputs/sequence_simulation
```

#### Recommend treatments with the inverse recommender

```bash
ipsc-twin recommend \
  --registry-dir state \
  --model-id <MODEL_ID> \
  --treatment-library examples/treatment_library.yaml \
  --current-adata data/current_cells.h5ad \
  --target-adata data/target_state_reference.h5ad \
  --top-n 10
```

### Limitations

- This is not official CPA unless an official CPA backend is separately added.
- The model requires sufficient perturbation and covariate coverage to generalize reliably.
- Predictions are experimental and require prospective validation.
- The current system is not clinical and must not be interpreted as a clinical decision system.

## How To Run

### 1. Create an experiment record

```bash
ipsc-twin registry create-experiment \
  --registry-dir state \
  --experiment-id exp_0001 \
  --name "Two-stage iPSC perturbation pilot" \
  --objective "Optimize sequential differentiation treatment combinations" \
  --owner "team_name"
```

### 2. List experiments

```bash
ipsc-twin registry list-experiments --registry-dir state
```

### 3. Register a dataset version

```bash
ipsc-twin registry register-dataset \
  --registry-dir state \
  --experiment-id exp_0001 \
  --anndata-path data/rounds_1_2_merged.h5ad \
  --preprocessing-version qc_v1_scanpy_v1 \
  --schema-version schema_v1
```

### 4. Train the baseline model

```bash
ipsc-twin train \
  --registry-dir state \
  --output-dir outputs/training \
  --dataset-id <DATASET_ID> \
  --model-type baseline \
  --random-seed 0
```

### 5. List registered models

```bash
ipsc-twin models list --registry-dir state
```

### 6. Describe a model

```bash
ipsc-twin models describe <MODEL_ID> --registry-dir state
```

## Typical Development Workflow

1. Create an experiment record for a new study.
2. Register each incoming AnnData dataset version with preprocessing and schema metadata.
3. Validate and update the treatment library before model training or reporting.
4. Train baseline models and register artifacts in the model registry.
5. Run validation using holdout strategies that match the intended generalization target.
6. Export dashboard artifacts for analysis, tracking, or UI work.
7. Generate the next DOE recommendation table for the next perturbation round.
8. Optionally synthesize literature and CMC notes to support treatment rationale review.

## Key Output Artifacts

### Registry State
These are typically stored under a local registry directory such as `state/`.
- `experiments.json`
- `datasets.json`
- `models.json`

### Training Outputs
Typical training output directory contents:
- model artifact, for example `baseline_<MODEL_ID>.pkl`
- metrics JSON
- model card Markdown

### Validation Outputs
Validation report helpers generate:
- `validation_metrics.json`
- `validation_predictions_vs_observed.csv`
- `validation_report.md`
- `regenai_pt_validation.json`
- `regenai_pt_validation.md`
- `regenai_pt_validation_metrics.csv`

### Dashboard Exports
The dashboard export layer writes:
- `experiment_summary.json`
- `qc_summary.json`
- `treatment_rankings.json`
- `treatment_rankings.csv`
- `clone_skew_summary.csv`
- `model_metrics.json`
- `next_doe.csv`
- `report.md`

### Literature Outputs
The optional literature module writes:
- `literature_evidence.json`
- `literature_synthesis.md`
- `treatment_rationale_table.csv`

## Treatment Library

An example treatment library is included at:
- `examples/treatment_library.yaml`

Each treatment entry is expected to include:
- `treatment_id`
- `treatment_name`
- `treatment_type`
- `pathway`
- `molecular_target`
- `dose`
- `dose_unit`
- `duration_hours`
- `vendor`
- `catalog_number`
- `grade`
- `mechanism_notes`
- `safety_notes`
- `cost_estimate`
- `active`

Allowed enums:
- `treatment_type`: `small_molecule`, `protein`, `cytokine`, `antibody`, `gene_editing`, `other`
- `grade`: `RUO`, `GMP`, `unknown`

When AnnData contains treatment names that are not present in the library, the mapper reports them and marks them as unknown so downstream review can catch gaps before production runs.

## Validation Framework

The production validation layer currently supports holdout strategies for:
- treatment
- iPSC line
- treatment sequence
- replicate

Implemented metrics include:
- Spearman correlation for predicted vs observed treatment ranking
- RMSE for condition-level scores
- top-k overlap
- direction accuracy relative to control
- stress-risk detection accuracy when labels are available

Use these to test both average predictive quality and the quality of experimental prioritization.

## Active Learning / DOE Recommendation

`recommend_next_doe(...)` selects the next set of conditions using four buckets:
- Exploitation: high predicted `final_score`
- Exploration: high uncertainty with acceptable predicted safety
- Diversity: broader mechanism/pathway coverage
- Controls: control, previous best, previous worst, and standard protocol when available

The output includes:
- DOE condition table
- replicate expansion
- plate map suggestion for two 48-well plates when possible
- Markdown rationale summary

## Optional Literature And CMC Module

This module is intentionally separate from the core modeling pipeline.

Design constraints:
- Do not send full papers in a single prompt
- Extract each paper or protocol text individually
- Synthesize compact structured summaries afterwards
- Use structured JSON outputs
- Use robust parsing for invalid JSON responses
- Enforce max character limits per paper chunk

Recommended usage pattern:
1. Prepare 5 to 10 short paper summaries, protocol excerpts, or curated notes.
2. Run paper-level extraction with an injected LLM client.
3. Run synthesis on the extracted evidence.
4. Save JSON, Markdown, and CSV outputs for scientist review.

## Development Checks

Run the standard local checks from the repository root:

```bash
pytest
ruff check src tests
mypy src
```

If you want a more explicit test run:

```bash
python -m pytest -q
```

## Mock / Local Pipeline Style Run

This Production repository does not replace the original MVP end-to-end pipeline, but it can support the same style of local iterative development.

A common local sequence is:
1. Register a mock or development dataset.
2. Train a baseline model.
3. Export dashboard artifacts.
4. Generate DOE recommendations.
5. Review validation and model-card outputs.

## CI

GitHub Actions workflow:
- `.github/workflows/ci.yml`

The workflow installs the package with dev dependencies and runs:
- `ruff check src tests`
- `mypy src`
- `pytest`

## Current Scope

### Production-grade today
- Local JSON-backed experiment, dataset, and model registries
- Reproducible baseline training records
- Lightweight treatment library management
- Validation metrics and reports
- DOE recommendation scaffolding
- Dashboard artifact export
- Optional literature/CMC evidence workflow

### Still evolving
- Cloud/object storage integration
- Automated ingestion from upstream wet-lab systems
- Scheduler/orchestrator integration
- Full production APIs and dashboard frontend
- Advanced perturbation models such as CPA-backed training
- Stronger access control and multi-user operations

## Related Documentation

Architecture design:
- `docs/production_architecture.md`

If you are extending the system, that design document is the best place to start before adding new services or changing registry and pipeline boundaries.
