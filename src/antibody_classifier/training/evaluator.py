"""
Model evaluation module.
Calculates comprehensive metrics for binary classification.
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class BinaryClassificationEvaluator:
    """
    Evaluator for binary classification tasks.
    Computes comprehensive metrics and provides detailed reports.
    """
    
    def __init__(self, positive_label: int = 1):
        """
        Initialise evaluator.
        
        Args:
            positive_label: Label value for positive class (binder)
        """
        self.positive_label = positive_label
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities for positive class (optional)
        
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=self.positive_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=self.positive_label, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=self.positive_label, zero_division=0)
        tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if y_pred_proba is not None:
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Could not compute AUROC: {e}")
                metrics['auroc'] = 0.0
            
            try:
                metrics['auprc'] = average_precision_score(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Could not compute AUPRC: {e}")
                metrics['auprc'] = 0.0
        
        return metrics
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        split_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model predictions with logging.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            split_name: Name of dataset split for logging
        
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on {split_name} set ({len(y_true):,} samples)")
        metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)
        logger.info(f"{split_name.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        logger.info(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        return metrics
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: list = None
    ) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: List of class names
        
        Returns:
            Classification report string
        """
        if target_names is None:
            target_names = ['non-binder', 'binder']
        
        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4
        )
        
        return report
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'auroc'
    ) -> Tuple[str, float]:
        """
        Compare multiple models and find the best one.
        
        Args:
            results: Dictionary of model_name -> metrics dict
            metric: Metric to use for comparison
        
        Returns:
            Tuple of (best_model_name, best_metric_value)
        """
        logger.info(f"Comparing models based on {metric}")
        
        best_model = None
        best_value = -np.inf
        for model_name, metrics in results.items():
            value = metrics.get(metric, -np.inf)
            logger.info(f"  {model_name}: {metric}={value:.4f}")
            if value > best_value:
                best_value = value
                best_model = model_name
        
        logger.info(f"\nBest model: {best_model} with {metric}={best_value:.4f}")
        
        return best_model, best_value


class LossTracker:
    """Track loss values during training."""
    
    def __init__(self):
        """Initialise loss tracker."""
        self.losses = []
        self.running_loss = 0.0
        self.n_batches = 0
    
    def update(self, loss: float, batch_size: int = 1):
        """
        Update with a new loss value.
        
        Args:
            loss: Loss value
            batch_size: Size of batch (for weighted averaging)
        """
        self.running_loss += loss * batch_size
        self.n_batches += batch_size
    
    def get_average(self) -> float:
        """Get average loss."""
        if self.n_batches == 0:
            return 0.0
        return self.running_loss / self.n_batches
    
    def reset(self):
        """Reset tracker."""
        self.running_loss = 0.0
        self.n_batches = 0
    
    def record_epoch(self):
        """Record current epoch average and reset."""
        avg_loss = self.get_average()
        self.losses.append(avg_loss)
        self.reset()
        return avg_loss


class EarlyStopping:
    """
    Early stopping to stop training when validation metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialise early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' to maximise metric, 'min' to minimise
            verbose: Whether to log messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        self.logger = setup_logger(__name__) if verbose else None
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current metric value
            epoch: Current epoch number
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # Check if score improved
        if self.mode == 'max':
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                self.logger.info(
                    f"Metric improved to {score:.4f} at epoch {epoch}"
                )
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(
                    f"No improvement for {self.counter}/{self.patience} epochs "
                    f"(best: {self.best_score:.4f} at epoch {self.best_epoch})"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.logger.info(
                        f"Early stopping triggered. Best score: {self.best_score:.4f} "
                        f"at epoch {self.best_epoch}"
                    )
        
        return self.early_stop
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best score."""
        return self.best_epoch