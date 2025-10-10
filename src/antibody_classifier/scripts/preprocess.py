"""
Data preprocessing script - CLI wrapper.
Thin wrapper around data.preprocessing module.
"""
import argparse
import sys
from pathlib import Path

if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from antibody_classifier.data.preprocessing import AntibodyDataProcessor
from antibody_classifier.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess antibody data: aggregate, label, and split'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw CSV data file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--binder-threshold',
        type=float,
        default=3.0,
        help='logKD threshold for binders (< threshold) (default: 3.0)'
    )
    parser.add_argument(
        '--non-binder-threshold',
        type=float,
        default=4.0,
        help='logKD threshold for non-binders (>= threshold) (default: 4.0)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='Test set ratio (default: 0.1)'
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
    log_file = log_dir / f'preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logger(__name__, str(log_file))
    
    logger.info("ANTIBODY DATA PREPROCESSING")
    
    processor = AntibodyDataProcessor(
        binder_threshold=args.binder_threshold,
        non_binder_threshold=args.non_binder_threshold,
        random_seed=args.random_seed
    )
    
    processor.process_pipeline(
        raw_data_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    logger.info("PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()