# SARS-CoV-2 Antibody Binding Classifier

Predict antibody binding to SARS-CoV-2 antigens from sequence alone using ESM-2 embeddings and gradient boosting.

## Setup

```bash
conda env create -f environment.yml
conda activate antibody_classifier
```

## Usage

Run scripts in order:

```bash
# 1. Preprocess data (aggregate, label, split)
./scripts/run_preprocess.sh [--input PATH] [--output-dir PATH]

# 2. Generate embeddings
./scripts/run_extract_embeddings.sh [--data-dir PATH] [--output-dir PATH]

# 3. Train models
./scripts/run_train.sh [--embeddings-dir PATH] [--output-dir PATH]
```

Results saved to `outputs/` including models, metrics, and plots.
