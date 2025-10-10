#!/bin/bash

set -e  # Exit on error

# Default values
INPUT_DATA="data/raw/MITLL_AAlphaBio_Ab_Binding_dataset.csv"
OUTPUT_DIR="data/processed"
BINDER_THRESHOLD=3.0
NON_BINDER_THRESHOLD=4.0
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
RANDOM_SEED=42
LOG_DIR="logs"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input PATH              Path to raw CSV data file (default: $INPUT_DATA)"
    echo "  --output-dir PATH         Directory to save processed data (default: $OUTPUT_DIR)"
    echo "  --binder-threshold FLOAT  logKD threshold for binders (default: $BINDER_THRESHOLD)"
    echo "  --non-binder-threshold FLOAT  logKD threshold for non-binders (default: $NON_BINDER_THRESHOLD)"
    echo "  --train-ratio FLOAT       Training set ratio (default: $TRAIN_RATIO)"
    echo "  --val-ratio FLOAT         Validation set ratio (default: $VAL_RATIO)"
    echo "  --test-ratio FLOAT        Test set ratio (default: $TEST_RATIO)"
    echo "  --random-seed INT         Random seed (default: $RANDOM_SEED)"
    echo "  --log-dir PATH            Directory for log files (default: $LOG_DIR)"
    echo "  -h, --help                Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DATA="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --binder-threshold)
            BINDER_THRESHOLD="$2"
            shift 2
            ;;
        --non-binder-threshold)
            NON_BINDER_THRESHOLD="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --val-ratio)
            VAL_RATIO="$2"
            shift 2
            ;;
        --test-ratio)
            TEST_RATIO="$2"
            shift 2
            ;;
        --random-seed)
            RANDOM_SEED="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

if [ ! -f "$INPUT_DATA" ]; then
    echo "Error: Input file not found: $INPUT_DATA"
    exit 1
fi

echo "========================================================================"
echo "DATA PREPROCESSING"
echo "========================================================================"
echo "Input data: $INPUT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "Binder threshold: < $BINDER_THRESHOLD"
echo "Non-binder threshold: >= $NON_BINDER_THRESHOLD"
echo "Split ratios: $TRAIN_RATIO / $VAL_RATIO / $TEST_RATIO"
echo "Random seed: $RANDOM_SEED"
echo "Log directory: $LOG_DIR"
echo "========================================================================"
echo ""

PYTHONPATH=src python -m antibody_classifier.scripts.preprocess \
    --input "$INPUT_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --binder-threshold "$BINDER_THRESHOLD" \
    --non-binder-threshold "$NON_BINDER_THRESHOLD" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO" \
    --random-seed "$RANDOM_SEED" \
    --log-dir "$LOG_DIR"

echo ""
echo "Preprocessing complete!"
echo "Next step: Run scripts/run_extract_embeddings.sh"