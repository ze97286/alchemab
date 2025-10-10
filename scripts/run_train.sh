#!/bin/bash

set -e  # Exit on error

# Default values
EMBEDDINGS_DIR="data/embeddings"
OUTPUT_DIR="outputs"
MODELS="xgboost lightgbm"
NUM_BOOST_ROUND=500
EARLY_STOPPING_ROUNDS=50
LEARNING_RATE=0.05
MAX_DEPTH=6
MIN_CHILD_WEIGHT=1
SUBSAMPLE=0.8
COLSAMPLE_BYTREE=0.8
REG_ALPHA=0.1
REG_LAMBDA=1.0
RANDOM_SEED=42
LOG_DIR="logs"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --embeddings-dir PATH         Directory with embedding files (default: $EMBEDDINGS_DIR)"
    echo "  --output-dir PATH             Directory to save models and results (default: $OUTPUT_DIR)"
    echo "  --models MODEL [MODEL ...]    Models to train (default: $MODELS)"
    echo "                                Choices: xgboost, lightgbm"
    echo "  --num-boost-round INT         Maximum boosting rounds (default: $NUM_BOOST_ROUND)"
    echo "  --early-stopping-rounds INT   Early stopping rounds (default: $EARLY_STOPPING_ROUNDS)"
    echo "  --learning-rate FLOAT         Learning rate (default: $LEARNING_RATE)"
    echo "  --max-depth INT               Maximum tree depth (default: $MAX_DEPTH)"
    echo "  --min-child-weight FLOAT      Minimum child weight for regularization (default: $MIN_CHILD_WEIGHT)"
    echo "  --subsample FLOAT             Subsample ratio of training instances (default: $SUBSAMPLE)"
    echo "  --colsample-bytree FLOAT      Subsample ratio of features (default: $COLSAMPLE_BYTREE)"
    echo "  --reg-alpha FLOAT             L1 regularization (default: $REG_ALPHA)"
    echo "  --reg-lambda FLOAT            L2 regularization (default: $REG_LAMBDA)"
    echo "  --random-seed INT             Random seed for reproducibility (default: $RANDOM_SEED)"
    echo "  --log-dir PATH                Directory for log files (default: $LOG_DIR)"
    echo "  -h, --help                    Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --embeddings-dir)
            EMBEDDINGS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models)
            MODELS=""
            shift
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                MODELS="$MODELS $1"
                shift
            done
            ;;
        --num-boost-round)
            NUM_BOOST_ROUND="$2"
            shift 2
            ;;
        --early-stopping-rounds)
            EARLY_STOPPING_ROUNDS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --min-child-weight)
            MIN_CHILD_WEIGHT="$2"
            shift 2
            ;;
        --subsample)
            SUBSAMPLE="$2"
            shift 2
            ;;
        --colsample-bytree)
            COLSAMPLE_BYTREE="$2"
            shift 2
            ;;
        --reg-alpha)
            REG_ALPHA="$2"
            shift 2
            ;;
        --reg-lambda)
            REG_LAMBDA="$2"
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

# Check if embeddings directory exists
if [ ! -d "$EMBEDDINGS_DIR" ]; then
    echo "Error: Embeddings directory not found: $EMBEDDINGS_DIR"
    echo "Have you run scripts/run_extract_embeddings.sh first?"
    exit 1
fi

# Check for required files
for split in train val test; do
    if [ ! -f "$EMBEDDINGS_DIR/${split}_embeddings.npz" ]; then
        echo "Error: Missing file: $EMBEDDINGS_DIR/${split}_embeddings.npz"
        echo "Have you run scripts/run_extract_embeddings.sh first?"
        exit 1
    fi
done

# Print configuration
echo "========================================================================"
echo "MODEL TRAINING AND EVALUATION"
echo "========================================================================"
echo "Embeddings directory: $EMBEDDINGS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Models: $MODELS"
echo "Num boost rounds: $NUM_BOOST_ROUND"
echo "Early stopping rounds: $EARLY_STOPPING_ROUNDS"
echo "Learning rate: $LEARNING_RATE"
echo "Max depth: $MAX_DEPTH"
echo "Min child weight: $MIN_CHILD_WEIGHT"
echo "Subsample: $SUBSAMPLE"
echo "Colsample bytree: $COLSAMPLE_BYTREE"
echo "Reg alpha (L1): $REG_ALPHA"
echo "Reg lambda (L2): $REG_LAMBDA"
echo "Random seed: $RANDOM_SEED"
echo "Log directory: $LOG_DIR"
echo "========================================================================"
echo ""

# Run Python script using -m
PYTHONPATH=src python -m antibody_classifier.scripts.train \
    --embeddings-dir "$EMBEDDINGS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --models $MODELS \
    --num-boost-round "$NUM_BOOST_ROUND" \
    --early-stopping-rounds "$EARLY_STOPPING_ROUNDS" \
    --learning-rate "$LEARNING_RATE" \
    --max-depth "$MAX_DEPTH" \
    --min-child-weight "$MIN_CHILD_WEIGHT" \
    --subsample "$SUBSAMPLE" \
    --colsample-bytree "$COLSAMPLE_BYTREE" \
    --reg-alpha "$REG_ALPHA" \
    --reg-lambda "$REG_LAMBDA" \
    --random-seed "$RANDOM_SEED" \
    --log-dir "$LOG_DIR"

echo ""
echo "Training complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Plots saved to: $OUTPUT_DIR/plots"