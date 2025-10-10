"""
Logging utilities for the antibody classifier.
Provides consistent logging across all modules.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if logger.handlers:
        return logger
    
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    formatter = logging.Formatter(
        format_string,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_timestamp() -> str:
    """Get timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ProgressLogger:
    """
    Logger for tracking progress with periodic updates.
    Useful for long-running operations like embedding extraction.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        total: int,
        desc: str = "Processing",
        log_interval: int = 100
    ):
        """
        Initialise progress logger.
        
        Args:
            logger: Logger instance to use
            total: Total number of items to process
            desc: Description of the operation
            log_interval: Log progress every N items
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.current = 0
        self.start_time = datetime.now()
    
    def update(self, n: int = 1):
        """
        Update progress counter.
        
        Args:
            n: Number of items processed
        """
        self.current += n
        
        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            progress_pct = (self.current / self.total) * 100
            items_per_sec = self.current / elapsed if elapsed > 0 else 0
            
            if self.current < self.total:
                eta_seconds = (self.total - self.current) / items_per_sec if items_per_sec > 0 else 0
                eta_str = f"ETA: {eta_seconds:.0f}s"
            else:
                eta_str = f"Total time: {elapsed:.0f}s"
            
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} "
                f"({progress_pct:.1f}%) - "
                f"{items_per_sec:.2f} items/s - "
                f"{eta_str}"
            )
    
    def close(self):
        """Log final completion message."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(
            f"{self.desc} completed: {self.total} items in {elapsed:.2f}s "
            f"({self.total/elapsed:.2f} items/s)"
        )


class MetricsLogger:
    """Logger for tracking training metrics across epochs."""
    
    def __init__(self, logger: logging.Logger, metrics: tuple):
        """
        Initialise metrics logger.
        
        Args:
            logger: Logger instance to use
            metrics: Tuple of metric names to track
        """
        self.logger = logger
        self.metrics = metrics
        self.history = {
            "train": {metric: [] for metric in metrics},
            "val": {metric: [] for metric in metrics}
        }
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        lr: Optional[float] = None
    ):
        """
        Log metrics for a training epoch.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            lr: Current learning rate (optional)
        """
        # Store metrics
        for metric in self.metrics:
            if metric in train_metrics:
                self.history["train"][metric].append(train_metrics[metric])
            if metric in val_metrics:
                self.history["val"][metric].append(val_metrics[metric])
        
        # Format log message
        train_str = " | ".join([
            f"train_{k}: {v:.4f}" for k, v in train_metrics.items()
        ])
        val_str = " | ".join([
            f"val_{k}: {v:.4f}" for k, v in val_metrics.items()
        ])
        
        lr_str = f" | lr: {lr:.6f}" if lr is not None else ""
        
        self.logger.info(
            f"Epoch {epoch:3d} | {train_str} | {val_str}{lr_str}"
        )
    
    def log_best_epoch(self, epoch: int, metric: str, value: float):
        """
        Log information about best epoch.
        
        Args:
            epoch: Best epoch number
            metric: Metric that was optimised
            value: Best metric value
        """
        self.logger.info(
            f"Best model at epoch {epoch} with {metric}: {value:.4f}"
        )
    
    def get_history(self) -> dict:
        """Get complete training history."""
        return self.history