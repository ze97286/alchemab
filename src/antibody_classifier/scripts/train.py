"""
Training and evaluation script - CLI wrapper.
Thin wrapper around models and training modules.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path if needed
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from antibody_classifier.models.xgboost_classifier import XGBoostClassifier
from antibody_classifier.models.lightgbm_classifier import LightGBMClassifier
from antibody_classifier.training.evaluator import BinaryClassificationEvaluator
from antibody_classifier.utils.logger import setup_logger
from antibody_classifier.utils.plotting import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_model_comparison,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_feature_importance,
    plot_calibration_curve,
    plot_threshold_analysis
)


def load_embeddings(embeddings_dir, logger):
    """Load embeddings for all splits."""
    logger.info(f"Loading embeddings from {embeddings_dir}")
    
    embeddings_path = Path(embeddings_dir)
    data = {}
    
    for split_name in ['train', 'val', 'test']:
        filepath = embeddings_path / f'{split_name}_embeddings.npz'
        
        if not filepath.exists():
            raise FileNotFoundError(f"Embeddings not found: {filepath}")
        
        npz = np.load(filepath)
        data[split_name] = {
            'X': npz['embeddings'],
            'y': npz['labels'],
            'ids': npz['ids']
        }
        logger.info(f"Loaded {split_name}: X shape {data[split_name]['X'].shape}")
    
    return data


def calculate_class_weight(y_train, logger):
    """Calculate class weight for imbalanced data."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    weight = n_neg / n_pos
    
    logger.info(f"Class distribution: {n_neg:,} non-binders, {n_pos:,} binders")
    logger.info(f"Calculated class weight: {weight:.3f}")
    
    return weight


def evaluate_and_log(model, X, y, split_name, logger, evaluator):
    """Evaluate model and log results."""
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    metrics = evaluator.evaluate(y, y_pred, y_pred_proba, split_name)
    return y_pred, y_pred_proba, metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate antibody binding classifiers'
    )
    parser.add_argument(
        '--embeddings-dir',
        type=str,
        required=True,
        help='Directory containing embedding files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save models and results'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['xgboost', 'lightgbm'],
        choices=['xgboost', 'lightgbm'],
        help='Models to train (default: xgboost lightgbm)'
    )
    parser.add_argument(
        '--num-boost-round',
        type=int,
        default=500,
        help='Maximum number of boosting rounds (default: 500)'
    )
    parser.add_argument(
        '--early-stopping-rounds',
        type=int,
        default=50,
        help='Early stopping rounds (default: 50)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.05,
        help='Learning rate (default: 0.05)'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=6,
        help='Maximum tree depth (default: 6)'
    )
    parser.add_argument(
        '--min-child-weight',
        type=float,
        default=1,
        help='Minimum child weight for regularization (default: 1)'
    )
    parser.add_argument(
        '--subsample',
        type=float,
        default=0.8,
        help='Subsample ratio of training instances (default: 0.8)'
    )
    parser.add_argument(
        '--colsample-bytree',
        type=float,
        default=0.8,
        help='Subsample ratio of features (default: 0.8)'
    )
    parser.add_argument(
        '--reg-alpha',
        type=float,
        default=0.1,
        help='L1 regularization (default: 0.1)'
    )
    parser.add_argument(
        '--reg-lambda',
        type=float,
        default=1.0,
        help='L2 regularization (default: 1.0)'
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
    
    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logger = setup_logger(__name__, str(log_file))
    
    logger.info("MODEL TRAINING AND EVALUATION")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load embeddings
    data = load_embeddings(args.embeddings_dir, logger)
    
    X_train, y_train = data['train']['X'], data['train']['y']
    X_val, y_val = data['val']['X'], data['val']['y']
    X_test, y_test = data['test']['X'], data['test']['y']
    
    logger.info(f"\nData shapes:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    logger.info(f"  Test: {X_test.shape}")
    
    # Calculate class weight
    class_weight = calculate_class_weight(y_train, logger)
    
    # Initialise evaluator
    evaluator = BinaryClassificationEvaluator()
    
    # Store results
    all_results = []
    
    # Train models
    if 'xgboost' in args.models:
        logger.info("TRAINING XGBOOST")
        
        # Initialise model
        xgb_model = XGBoostClassifier(
            n_estimators=args.num_boost_round,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            scale_pos_weight=class_weight,
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.random_seed
        )
        
        # Train
        history = xgb_model.train(X_train, y_train, X_val, y_val, verbose=True)
        
        # Evaluate
        logger.info("\nEvaluating XGBoost...")
        train_pred, train_proba, train_metrics = evaluate_and_log(
            xgb_model, X_train, y_train, "Train", logger, evaluator
        )
        val_pred, val_proba, val_metrics = evaluate_and_log(
            xgb_model, X_val, y_val, "Validation", logger, evaluator
        )
        test_pred, test_proba, test_metrics = evaluate_and_log(
            xgb_model, X_test, y_test, "Test", logger, evaluator
        )
        
        # Save plots
        xgb_plot_dir = plots_dir / 'xgboost'
        xgb_plot_dir.mkdir(exist_ok=True)

        plot_training_curves(
            history,
            output_path=str(xgb_plot_dir / 'training_curves.html'),
            title='XGBoost - Training Curves'
        )
        plot_confusion_matrix(
            y_test,
            test_pred,
            ['non-binder', 'binder'],
            output_path=str(xgb_plot_dir / 'confusion_matrix.html'),
            title='XGBoost - Test Set Confusion Matrix'
        )
        plot_roc_curve(
            y_test,
            test_proba,
            output_path=str(xgb_plot_dir / 'roc_curve.html'),
            title='XGBoost - ROC Curve',
            y_true_val=y_val,
            y_pred_proba_val=val_proba
        )
        plot_precision_recall_curve(
            y_test,
            test_proba,
            output_path=str(xgb_plot_dir / 'precision_recall_curve.html'),
            title='XGBoost - Precision-Recall Curve',
            y_true_val=y_val,
            y_pred_proba_val=val_proba
        )
        plot_calibration_curve(
            y_test,
            test_proba,
            output_path=str(xgb_plot_dir / 'calibration_curve.html'),
            title='XGBoost - Calibration Curve'
        )
        plot_threshold_analysis(
            y_test,
            test_proba,
            output_path=str(xgb_plot_dir / 'threshold_analysis.html'),
            title='XGBoost - Threshold Analysis'
        )

        # Feature importance
        feature_indices, importance_scores = xgb_model.get_feature_importance()
        feature_names = [f'dim_{i}' for i in feature_indices]
        plot_feature_importance(
            feature_names,
            importance_scores,
            output_path=str(xgb_plot_dir / 'feature_importance.html'),
            title='XGBoost - Feature Importance',
            top_n=50
        )
        
        # Save model
        xgb_model.save(str(output_dir / 'xgboost_model'))
        logger.info(f"Saved model: {output_dir / 'xgboost_model.json'}")
        
        # Store results
        all_results.append({
            'model': 'XGBoost',
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()}
        })
    
    if 'lightgbm' in args.models:
        logger.info("TRAINING LIGHTGBM")
        
        # Initialise model
        lgb_model = LightGBMClassifier(
            n_estimators=args.num_boost_round,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            scale_pos_weight=class_weight,
            early_stopping_rounds=args.early_stopping_rounds,
            random_state=args.random_seed
        )
        
        # Train
        history = lgb_model.train(X_train, y_train, X_val, y_val, verbose=True)
        
        # Evaluate
        logger.info("\nEvaluating LightGBM...")
        train_pred, train_proba, train_metrics = evaluate_and_log(
            lgb_model, X_train, y_train, "Train", logger, evaluator
        )
        val_pred, val_proba, val_metrics = evaluate_and_log(
            lgb_model, X_val, y_val, "Validation", logger, evaluator
        )
        test_pred, test_proba, test_metrics = evaluate_and_log(
            lgb_model, X_test, y_test, "Test", logger, evaluator
        )
        
        # Save plots
        lgb_plot_dir = plots_dir / 'lightgbm'
        lgb_plot_dir.mkdir(exist_ok=True)

        plot_training_curves(
            history,
            output_path=str(lgb_plot_dir / 'training_curves.html'),
            title='LightGBM - Training Curves'
        )
        plot_confusion_matrix(
            y_test,
            test_pred,
            ['non-binder', 'binder'],
            output_path=str(lgb_plot_dir / 'confusion_matrix.html'),
            title='LightGBM - Test Set Confusion Matrix'
        )
        plot_roc_curve(
            y_test,
            test_proba,
            output_path=str(lgb_plot_dir / 'roc_curve.html'),
            title='LightGBM - ROC Curve',
            y_true_val=y_val,
            y_pred_proba_val=val_proba
        )
        plot_precision_recall_curve(
            y_test,
            test_proba,
            output_path=str(lgb_plot_dir / 'precision_recall_curve.html'),
            title='LightGBM - Precision-Recall Curve',
            y_true_val=y_val,
            y_pred_proba_val=val_proba
        )
        plot_calibration_curve(
            y_test,
            test_proba,
            output_path=str(lgb_plot_dir / 'calibration_curve.html'),
            title='LightGBM - Calibration Curve'
        )
        plot_threshold_analysis(
            y_test,
            test_proba,
            output_path=str(lgb_plot_dir / 'threshold_analysis.html'),
            title='LightGBM - Threshold Analysis'
        )

        feature_indices, importance_scores = lgb_model.get_feature_importance()
        feature_names = [f'dim_{i}' for i in feature_indices]
        plot_feature_importance(
            feature_names,
            importance_scores,
            output_path=str(lgb_plot_dir / 'feature_importance.html'),
            title='LightGBM - Feature Importance',
            top_n=50
        )
        
        lgb_model.save(str(output_dir / 'lightgbm_model'))
        logger.info(f"Saved model: {output_dir / 'lightgbm_model.txt'}")
        
        all_results.append({
            'model': 'LightGBM',
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()}
        })
    
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / 'results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nResults saved to: {results_path}")
    
    logger.info("MODEL COMPARISON - TEST SET")
    comparison_cols = ['model', 'test_accuracy', 'test_auroc', 'test_auprc', 'test_f1']
    logger.info("\n" + results_df[comparison_cols].to_string(index=False))
    
    for metric in ['auroc', 'auprc', 'f1', 'accuracy']:
        plot_model_comparison(
            results_df,
            metric=metric,
            output_path=str(plots_dir / f'comparison_{metric}.html'),
            title=f'Model Comparison - {metric.upper()}'
        )
    
    logger.info("TRAINING COMPLETE")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()