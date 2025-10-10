"""
XGBoost classifier for antibody binding prediction.
"""
import xgboost as xgb
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class XGBoostClassifier:
    """
    XGBoost binary classifier with early stopping and evaluation.
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: float = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 50,
        eval_metric: str = "auc",
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Initialise XGBoost classifier.

        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            min_child_weight: Minimum sum of instance weight needed in a child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing trees
            reg_alpha: L1 regularisation term
            reg_lambda: L2 regularisation term
            scale_pos_weight: Balancing of positive and negative weights
            early_stopping_rounds: Rounds without improvement before stopping
            eval_metric: Evaluation metric for early stopping
            n_jobs: Number of parallel threads (-1 = all cores)
            random_state: Random seed
        """
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': eval_metric,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'tree_method': 'hist'  # Faster for CPU
        }
        
        if scale_pos_weight is not None:
            self.params['scale_pos_weight'] = scale_pos_weight
        
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.best_iteration = None
        
        logger.info("Initialised XGBoost classifier")
        logger.info(f"Parameters: {self.params}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            verbose: Whether to print training progress
        
        Returns:
            Dictionary of training history
        """
        logger.info(f"Training XGBoost classifier on {len(X_train):,} samples")
        logger.info(f"Validation set: {len(X_val):,} samples")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose
        )
        
        self.best_iteration = self.model.best_iteration
        
        logger.info(f"Training completed")
        logger.info(f"Best iteration: {self.best_iteration}")
        logger.info(f"Best validation {self.params['eval_metric']}: "
                   f"{self.model.best_score:.4f}")
        
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
        
        dmatrix = xgb.DMatrix(X)
        probas = self.model.predict(dmatrix)
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
        
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)
    
    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
        
        Returns:
            Tuple of (feature indices, importance scores)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        features = []
        scores = []
        for feat, score in importance_dict.items():
            feat_idx = int(feat.replace('f', ''))
            features.append(feat_idx)
            scores.append(score)
        
        sorted_indices = np.argsort(scores)[::-1]
        features = np.array(features)[sorted_indices]
        scores = np.array(scores)[sorted_indices]
        
        return features, scores
    
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
        
        model_path = save_path.with_suffix('.json')
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
        model_path = load_path.with_suffix('.json')
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        params_path = load_path.with_suffix('.pkl')
        with open(params_path, 'rb') as f:
            saved_data = pickle.load(f)
            self.params = saved_data['params']
            self.n_estimators = saved_data['n_estimators']
            self.early_stopping_rounds = saved_data['early_stopping_rounds']
            self.best_iteration = saved_data['best_iteration']
        
        logger.info(f"Model loaded from {model_path}")