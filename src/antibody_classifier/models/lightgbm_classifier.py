"""
LightGBM classifier for antibody binding prediction.
"""
import lightgbm as lgb
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class LightGBMClassifier:
    """
    LightGBM binary classifier with early stopping and evaluation.
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: float = 1,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 50,
        metric: str = "auc",
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialise LightGBM classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            min_child_weight: Minimum sum of instance weight needed in a child
            num_leaves: Maximum number of leaves in one tree
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing trees
            reg_alpha: L1 regularisation term
            reg_lambda: L2 regularisation term
            scale_pos_weight: Balancing of positive and negative weights
            early_stopping_rounds: Rounds without improvement before stopping
            metric: Evaluation metric for early stopping
            n_jobs: Number of parallel threads (-1 = all cores)
            random_state: Random seed
        """
        self.params = {
            'objective': 'binary',
            'metric': metric,
            'boosting_type': 'gbdt',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'num_leaves': num_leaves,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'verbose': -1
        }
        
        if scale_pos_weight is not None:
            self.params['scale_pos_weight'] = scale_pos_weight
        
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.best_iteration = None
        
        logger.info("Initialised LightGBM classifier")
        logger.info(f"Parameters: {self.params}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Dict[str, list]]:
        """
        Train LightGBM model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print training progress
        
        Returns:
            Dictionary of training history
        """
        logger.info(f"Training LightGBM classifier on {len(X_train):,} samples")
        logger.info(f"Validation set: {len(X_val):,} samples")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        evals_result = {}
        
        callbacks = [
            lgb.log_evaluation(period=10 if verbose else 0),
            lgb.record_evaluation(evals_result)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        self.best_iteration = self.model.best_iteration
        
        logger.info(f"Training completed")
        logger.info(f"Best iteration: {self.best_iteration}")
        
        if self.params['metric'] in evals_result['val']:
            best_score = evals_result['val'][self.params['metric']][self.best_iteration - 1]
            logger.info(f"Best validation {self.params['metric']}: {best_score:.4f}")
        
        return evals_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
        
        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        probas = self.model.predict(
            X,
            num_iteration=self.best_iteration
        )
        return (probas >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Predicted probabilities for positive class
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(
            X,
            num_iteration=self.best_iteration
        )
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('split' or 'gain')
        
        Returns:
            Tuple of (feature indices, importance scores)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_indices = np.arange(len(importance))
        sorted_indices = np.argsort(importance)[::-1]
        
        return feature_indices[sorted_indices], importance[sorted_indices]
    
    def save(self, filepath: str):
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_path = save_path.with_suffix('.txt')
        self.model.save_model(str(model_path))
        params_path = save_path.with_suffix('.pkl')
        with open(params_path, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'n_estimators': self.n_estimators,
                'early_stopping_rounds': self.early_stopping_rounds,
                'best_iteration': self.best_iteration
            }, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from
        """
        load_path = Path(filepath)
        model_path = load_path.with_suffix('.txt')
        self.model = lgb.Booster(model_file=str(model_path))
        params_path = load_path.with_suffix('.pkl')
        with open(params_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.params = saved_data['params']
            self.n_estimators = saved_data['n_estimators']
            self.early_stopping_rounds = saved_data['early_stopping_rounds']
            self.best_iteration = saved_data['best_iteration']
        
        logger.info(f"Model loaded from {model_path}")