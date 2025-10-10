"""
ESM-2 embedding extraction script - CLI wrapper.
Thin wrapper around data.embeddings module.
"""
import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from antibody_classifier.data.embeddings import extract_and_save_embeddings
from antibody_classifier.data.preprocessing import load_processed_data
from antibody_classifier.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description='Extract ESM-2 embeddings from antibody sequences'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing processed CSV files (train.csv, val.csv, test.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save embedding files'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='esm2_t33_650M_UR50D',
        choices=['esm2_t30_150M_UR50D', 'esm2_t33_650M_UR50D', 'esm2_t36_3B_UR50D'],
        help='ESM-2 model variant (default: esm2_t33_650M_UR50D)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=None,
        help='Number of CPU threads to use (default: use all available)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory to save log files (default: logs)'
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f'embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logger(__name__, str(log_file))
    
    logger.info("ESM-2 EMBEDDING EXTRACTION")
    data_splits = load_processed_data(args.input_dir)
    
    extract_and_save_embeddings(
        data_splits=data_splits,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        random_seed=args.random_seed
    )
    
    logger.info("EMBEDDING EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()