#!/bin/bash

set -e  # Exit on error

# Default values
INPUT_DIR="data/processed"
OUTPUT_DIR="data/embeddings"
MODEL_NAME="esm2_t33_650M_UR50D"
BATCH_SIZE=8
NUM_THREADS=""
RANDOM_SEED=42
LOG_DIR="logs"

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-dir PATH          Directory with processed CSV files (default: $INPUT_DIR)"
    echo "  --output-dir PATH         Directory to save embeddings (default: $OUTPUT_DIR)"
    echo "  --model-name NAME         ESM-2 model variant (default: $MODEL_NAME)"
    echo "                            Choices: esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D"
    echo "  --batch-size INT          Batch size for processing (default: $BATCH_SIZE)"
    echo "  --num-threads INT         Number of CPU threads to use (default: use all available)"
    echo "  --random-seed INT         Random seed for reproducibility (default: $RANDOM_SEED)"
    echo "  --log-dir PATH            Directory for log files (default: $LOG_DIR)"
    echo "  -h, --help                Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-threads)
            NUM_THREADS="$2"
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

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo "Have you run scripts/run_preprocess.sh first?"
    exit 1
fi

for split in train val test; do
    if [ ! -f "$INPUT_DIR/${split}.csv" ]; then
        echo "Error: Missing file: $INPUT_DIR/${split}.csv"
        echo "Have you run scripts/run_preprocess.sh first?"
        exit 1
    fi
done

echo "========================================================================"
echo "ESM-2 EMBEDDING EXTRACTION"
echo "========================================================================"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
if [ -n "$NUM_THREADS" ]; then
    echo "CPU threads: $NUM_THREADS"
else
    echo "CPU threads: all available"
fi
echo "Random seed: $RANDOM_SEED"
echo "Log directory: $LOG_DIR"
echo "========================================================================"
echo ""

CMD="PYTHONPATH=src python -m antibody_classifier.scripts.extract_embeddings \
    --input-dir \"$INPUT_DIR\" \
    --output-dir \"$OUTPUT_DIR\" \
    --model-name \"$MODEL_NAME\" \
    --batch-size \"$BATCH_SIZE\" \
    --random-seed \"$RANDOM_SEED\" \
    --num-threads \"$NUM_THREADS\" \
    --log-dir \"$LOG_DIR\""

eval $CMD

echo ""
echo "Embedding extraction complete!"
echo "Next step: Run scripts/run_train.sh"