"""
Data preprocessing module for antibody sequences.
Handles loading, filtering, and splitting data.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class AntibodyDataProcessor:
    """
    Processor for antibody-antigen binding data.
    Handles loading, aggregation, filtering, and splitting.
    """
    
    def __init__(
        self,
        binder_threshold: float = 3.0,
        non_binder_threshold: float = 4.0,
        random_seed: int = 42
    ):
        """
        Initialise data processor.
        
        Args:
            binder_threshold: logKD threshold for binders (< threshold)
            non_binder_threshold: logKD threshold for non-binders (>= threshold)
            random_seed: Random seed for reproducibility
        """
        self.binder_threshold = binder_threshold
        self.non_binder_threshold = non_binder_threshold
        self.random_seed = random_seed
        
        logger.info("Initialised AntibodyDataProcessor")
        logger.info(f"Binder threshold: logKD < {binder_threshold}")
        logger.info(f"Non-binder threshold: logKD >= {non_binder_threshold}")
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw antibody data from CSV.
        
        Args:
            filepath: Path to raw data CSV
        
        Returns:
            Raw dataframe
        """
        logger.info(f"Loading raw data from {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df):,} rows with {len(df.columns)} columns")
        logger.info(f"Unique antibodies (POI): {df['POI'].nunique():,}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df
    
    def aggregate_by_antibody(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate measurements by antibody, calculating mean logKD.
        
        Args:
            df: Raw dataframe with multiple measurements per antibody
        
        Returns:
            Aggregated dataframe with one row per antibody
        """
        logger.info("Aggregating measurements by antibody (POI)")
        
        # Group by POI and calculate stats
        aggregated = df.groupby('POI').agg({
            'Pred_affinity': ['mean', 'std', 'count'],
            'Sequence': 'first',
            'HC': 'first',
            'LC': 'first',
            'CDRH1': 'first',
            'CDRH2': 'first',
            'CDRH3': 'first',
            'CDRL1': 'first',
            'CDRL2': 'first',
            'CDRL3': 'first'
        }).reset_index()
        
        aggregated.columns = [
            'POI', 'mean_logKD', 'std_logKD', 'n_measurements',
            'Sequence', 'HC', 'LC', 'CDRH1', 'CDRH2', 'CDRH3',
            'CDRL1', 'CDRL2', 'CDRL3'
        ]
        
        logger.info(f"Aggregated to {len(aggregated):,} unique antibodies")
        logger.info(f"Antibodies with valid mean_logKD: {aggregated['mean_logKD'].notna().sum():,}")
        
        valid_logkd = aggregated['mean_logKD'].dropna()
        logger.info(f"Mean logKD statistics:")
        logger.info(f"  Mean: {valid_logkd.mean():.3f}")
        logger.info(f"  Std: {valid_logkd.std():.3f}")
        logger.info(f"  Min: {valid_logkd.min():.3f}")
        logger.info(f"  Max: {valid_logkd.max():.3f}")
        
        return aggregated
    
    def assign_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assign binder/non-binder classes based on thresholds.
        
        Args:
            df: Aggregated dataframe with mean_logKD
        
        Returns:
            Dataframe with 'class' column
        """
        logger.info("Assigning classes based on logKD thresholds")
        
        df = df.copy()
        
        df['class'] = 'ambiguous'
        df.loc[df['mean_logKD'] < self.binder_threshold, 'class'] = 'binder'
        df.loc[df['mean_logKD'] >= self.non_binder_threshold, 'class'] = 'non-binder'

        class_counts = df['class'].value_counts()
        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"  {class_name}: {count:,} ({pct:.2f}%)")
        
        return df
    
    def filter_ambiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ambiguous cases from dataset.
        
        Args:
            df: Dataframe with 'class' column
        
        Returns:
            Filtered dataframe
        """
        logger.info("Filtering ambiguous cases")
        
        initial_count = len(df)
        df_filtered = df[df['class'] != 'ambiguous'].copy()
        removed_count = initial_count - len(df_filtered)
        logger.info(f"Removed {removed_count:,} ambiguous cases")
        logger.info(f"Remaining: {len(df_filtered):,} antibodies")
        
        class_counts = df_filtered['class'].value_counts()
        logger.info("Final class distribution:")
        for class_name, count in class_counts.items():
            pct = (count / len(df_filtered)) * 100
            logger.info(f"  {class_name}: {count:,} ({pct:.2f}%)")
        
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}:1")
        
        return df_filtered
    
    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode class labels as integers.
        binder -> 1, non-binder -> 0
        
        Args:
            df: Dataframe with 'class' column
        
        Returns:
            Dataframe with 'label' column
        """
        df = df.copy()
        df['label'] = df['class'].map({'binder': 1, 'non-binder': 0})
        logger.info("Encoded labels: binder=1, non-binder=0")
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets with stratification.
        
        Args:
            df: Dataframe to split
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        logger.info(f"Splitting data: {train_ratio:.1%} train, {val_ratio:.1%} val, {test_ratio:.1%} test")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            stratify=df['label'],
            random_state=self.random_seed
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            stratify=temp_df['label'],
            random_state=self.random_seed
        )
        
        logger.info(f"Train set: {len(train_df):,} samples")
        self._log_split_distribution(train_df, "Train")
        
        logger.info(f"Validation set: {len(val_df):,} samples")
        self._log_split_distribution(val_df, "Validation")
        
        logger.info(f"Test set: {len(test_df):,} samples")
        self._log_split_distribution(test_df, "Test")
        
        return train_df, val_df, test_df
    
    def _log_split_distribution(self, df: pd.DataFrame, split_name: str):
        """Log class distribution for a split."""
        class_counts = df['class'].value_counts()
        for class_name, count in class_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"  {split_name} {class_name}: {count:,} ({pct:.2f}%)")
    
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str
    ):
        """
        Save processed datasets to disk.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed data to {output_dir}")
        
        train_df.to_csv(output_path / "train.csv", index=False)
        val_df.to_csv(output_path / "val.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        
        logger.info("Saved train.csv, val.csv, test.csv")
    
    def process_pipeline(
        self,
        raw_data_path: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, pd.DataFrame]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            raw_data_path: Path to raw CSV data
            output_dir: Directory to save processed data
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
        
        Returns:
            Dictionary with 'train', 'val', 'test' dataframes
        """
        logger.info("STARTING DATA PREPROCESSING PIPELINE")
        
        # Main processing steps:
        # 1. load raw data
        # 2. aggregate by antibody
        # 3. assign classes
        # 4. filter ambiguous
        # 5. encode labels
        # 6. split data and
        # 7. save processed data
        df = self.load_raw_data(raw_data_path)
        df_agg = self.aggregate_by_antibody(df)
        df_classified = self.assign_classes(df_agg)
        df_filtered = self.filter_ambiguous(df_classified)
        df_encoded = self.encode_labels(df_filtered)
        train_df, val_df, test_df = self.split_data(
            df_encoded,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        self.save_processed_data(train_df, val_df, test_df, output_dir)
        
        logger.info("DATA PREPROCESSING COMPLETE")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }


def load_processed_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load pre-processed data splits.
    
    Args:
        data_dir: Directory containing processed CSV files
    
    Returns:
        Dictionary with 'train', 'val', 'test' dataframes
    """
    logger.info(f"Loading processed data from {data_dir}")
    
    data_path = Path(data_dir)
    
    train_df = pd.read_csv(data_path / "train.csv")
    val_df = pd.read_csv(data_path / "val.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    
    logger.info(f"Loaded train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }