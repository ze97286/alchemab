"""
Plotting utilities using Plotly for visualisation.
All plots are interactive and saved as HTML files.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


def _save_figure(fig: go.Figure, output_path: str) -> None:
    """
    Save figure as both HTML and PNG.

    Args:
        fig: Plotly figure object
        output_path: Path to save HTML file (PNG will use same path with .png extension)
    """
    fig.write_html(output_path)
    # Also save as PNG
    png_path = output_path.replace('.html', '.png')
    try:
        fig.write_image(
            str(png_path),
            scale=2,
            width=1400,
            height=800,
            engine="kaleido"
        )
    except Exception:
        pass


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Optional[str] = None,
    title: str = "Class Distribution"
) -> go.Figure:
    """
    Plot class distribution as a bar chart.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        output_path: Path to save HTML file (optional)
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            text=list(class_counts.values()),
            textposition='auto',
            marker_color=['#2ecc71', '#e74c3c']
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Class",
        yaxis_title="Count",
        template="plotly_white",
        font=dict(size=12),
        height=500
    )
    
    if output_path:
        _save_figure(fig, output_path)
    
    return fig


def plot_training_curves(
    history: Dict[str, Dict[str, List[float]]],
    output_path: Optional[str] = None,
    title: str = "Training Curves"
) -> go.Figure:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with 'train' and 'val' subdicts containing metric lists
        output_path: Path to save HTML file (optional)
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    metrics = list(history['train'].keys())
    n_metrics = len(metrics)
    
    # Calculate subplot layout
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=metrics,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    for idx, metric in enumerate(metrics):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        epochs = list(range(1, len(history['train'][metric]) + 1))
        
        # Training curve
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history['train'][metric],
                mode='lines+markers',
                name=f'Train {metric}',
                line=dict(color='#3498db', width=2),
                marker=dict(size=4),
                showlegend=(idx == 0)
            ),
            row=row,
            col=col
        )
        
        # Validation curve
        if metric in history['val']:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=history['val'][metric],
                    mode='lines+markers',
                    name=f'Val {metric}',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=4),
                    showlegend=(idx == 0)
                ),
                row=row,
                col=col
            )
        
        fig.update_xaxes(title_text="Epoch", row=row, col=col)
        fig.update_yaxes(title_text=metric.capitalize(), row=row, col=col)
    
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=400 * n_rows,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    if output_path:
        _save_figure(fig, output_path)
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    title: str = "Confusion Matrix"
) -> go.Figure:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save HTML file (optional)
        title: Plot title
    
    Returns:
        Plotly figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalise confusion matrix
    cm_normalised = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{cm[i, j]}<br>({cm_normalised[i, j]:.2%})",
                    showarrow=False,
                    font=dict(color='white' if cm_normalised[i, j] > 0.5 else 'black')
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalised,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Normalised<br>Count")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        template="plotly_white",
        annotations=annotations,
        height=500,
        width=600
    )
    
    if output_path:
        _save_figure(fig, output_path)
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "ROC Curve",
    y_true_val: Optional[np.ndarray] = None,
    y_pred_proba_val: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot ROC curve with optimal threshold marked.

    Args:
        y_true: True binary labels for test set (0 or 1)
        y_pred_proba: Predicted probabilities for test set positive class
        output_path: Path to save HTML file (optional)
        title: Plot title
        y_true_val: True binary labels for validation set (optional)
        y_pred_proba_val: Predicted probabilities for validation set (optional)

    Returns:
        Plotly figure object
    """
    from sklearn.metrics import auc, f1_score, precision_score, accuracy_score

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    optimal_sensitivity = optimal_tpr  
    optimal_specificity = 1 - optimal_fpr  
    optimal_precision = precision_score(y_true, y_pred_optimal, zero_division=0)
    optimal_f1 = f1_score(y_true, y_pred_optimal, zero_division=0)
    optimal_accuracy = accuracy_score(y_true, y_pred_optimal)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'Test (AUC = {roc_auc:.3f})',
        line=dict(color='#3498db', width=2),
        hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))

    if y_true_val is not None and y_pred_proba_val is not None:
        fpr_val, tpr_val, thresholds_val = roc_curve(y_true_val, y_pred_proba_val)
        roc_auc_val = auc(fpr_val, tpr_val)
        fig.add_trace(go.Scatter(
            x=fpr_val,
            y=tpr_val,
            mode='lines',
            name=f'Validation (AUC = {roc_auc_val:.3f})',
            line=dict(color='#2ecc71', width=2),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random classifier',
        line=dict(color='gray', width=2, dash='dash'),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[optimal_fpr],
        y=[optimal_tpr],
        mode='markers',
        name=f'Optimal threshold = {optimal_threshold:.3f}',
        marker=dict(size=12, color='#e74c3c', symbol='star'),
        hovertemplate=f'<b>Optimal Threshold: {optimal_threshold:.3f}</b><br>' +
                     f'Sensitivity: {optimal_sensitivity:.3f}<br>' +
                     f'Specificity: {optimal_specificity:.3f}<br>' +
                     f'FPR: {optimal_fpr:.3f}<br>' +
                     f'TPR: {optimal_tpr:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=[optimal_fpr, optimal_fpr],
        y=[0, optimal_tpr],
        mode='lines',
        line=dict(color='#e74c3c', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[0, optimal_fpr],
        y=[optimal_tpr, optimal_tpr],
        mode='lines',
        line=dict(color='#e74c3c', width=1, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))

    annotation_text = (
        f"<b>Optimal Threshold: {optimal_threshold:.3f}</b><br>"
        f"Sensitivity: {optimal_sensitivity:.3f}<br>"
        f"Specificity: {optimal_specificity:.3f}<br>"
        f"Precision: {optimal_precision:.3f}<br>"
        f"F1 Score: {optimal_f1:.3f}<br>"
        f"Accuracy: {optimal_accuracy:.3f}"
    )

    fig.update_layout(
        title=f"{title}<br>AUROC = {roc_auc:.3f}",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        template="plotly_white",
        height=600,
        width=700,
        showlegend=True,
        annotations=[
            dict(
                x=0.98,
                y=0.02,
                xref='paper',
                yref='paper',
                text=annotation_text,
                showarrow=False,
                align='right',
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#e74c3c',
                borderwidth=2,
                borderpad=10,
                font=dict(size=11)
            )
        ]
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Precision-Recall Curve",
    y_true_val: Optional[np.ndarray] = None,
    y_pred_proba_val: Optional[np.ndarray] = None
) -> go.Figure:
    """
    Plot precision-recall curve.

    Args:
        y_true: True binary labels for test set (0 or 1)
        y_pred_proba: Predicted probabilities for test set positive class
        output_path: Path to save HTML file (optional)
        title: Plot title
        y_true_val: True binary labels for validation set (optional)
        y_pred_proba_val: Predicted probabilities for validation set (optional)

    Returns:
        Plotly figure object
    """
    from sklearn.metrics import average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'Test (AP = {ap:.3f})',
        line=dict(color='#e74c3c', width=2),
        fill='tozeroy'
    ))

    if y_true_val is not None and y_pred_proba_val is not None:
        precision_val, recall_val, _ = precision_recall_curve(y_true_val, y_pred_proba_val)
        ap_val = average_precision_score(y_true_val, y_pred_proba_val)
        fig.add_trace(go.Scatter(
            x=recall_val,
            y=precision_val,
            mode='lines',
            name=f'Validation (AP = {ap_val:.3f})',
            line=dict(color='#2ecc71', width=2)
        ))
    
    baseline = y_true.sum() / len(y_true)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode='lines',
        name=f'Baseline (prevalence = {baseline:.3f})',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"{title}<br>Average Precision = {ap:.3f}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template="plotly_white",
        height=500,
        width=600,
        showlegend=True
    )
    
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    
    if output_path:
        _save_figure(fig, output_path)
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_values: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Feature Importance",
    top_n: int = 20
) -> go.Figure:
    """
    Plot feature importance as a horizontal bar chart.

    Args:
        feature_names: List of feature names
        importance_values: Array of importance values
        output_path: Path to save HTML file (optional)
        title: Plot title
        top_n: Number of top features to show

    Returns:
        Plotly figure object
    """
    importance_values = np.array(importance_values)
    feature_names = np.array(feature_names)
    indices = np.argsort(importance_values)[::-1][:top_n]
    top_features = feature_names[indices].tolist()
    top_importance = importance_values[indices]
    
    fig = go.Figure(go.Bar(
        x=top_importance,
        y=top_features,
        orientation='h',
        marker_color='#9b59b6'
    ))
    
    fig.update_layout(
        title=f"{title} (Top {top_n})",
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(500, top_n * 25),
        yaxis=dict(autorange="reversed")
    )
    
    if output_path:
        _save_figure(fig, output_path)
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    output_path: Optional[str] = None,
    title: str = "Calibration Curve"
) -> go.Figure:
    """
    Plot calibration curve to assess probability calibration.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        output_path: Path to save HTML file (optional)
        title: Plot title

    Returns:
        Plotly figure object
    """
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode='lines+markers',
        name='Model',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect calibration',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        template="plotly_white",
        height=500,
        width=600,
        showlegend=True
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Optional[str] = None,
    title: str = "Threshold Analysis"
) -> go.Figure:
    """
    Plot precision, recall, and F1 at different classification thresholds.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        output_path: Path to save HTML file (optional)
        title: Plot title

    Returns:
        Plotly figure object
    """
    from sklearn.metrics import precision_recall_curve, f1_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)

    f1_scores = []
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precision[:-1],
        mode='lines',
        name='Precision',
        line=dict(color='#3498db', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recall[:-1],
        mode='lines',
        name='Recall',
        line=dict(color='#e74c3c', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=thresholds,
        y=f1_scores,
        mode='lines',
        name='F1 Score',
        line=dict(color='#2ecc71', width=2)
    ))

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    fig.add_trace(go.Scatter(
        x=[optimal_threshold],
        y=[optimal_f1],
        mode='markers',
        name=f'Optimal (thresh={optimal_threshold:.3f})',
        marker=dict(size=12, color='#9b59b6', symbol='star')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        template="plotly_white",
        height=500,
        width=800,
        showlegend=True
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    if output_path:
        _save_figure(fig, output_path)

    return fig


def plot_model_comparison(
    results_df: pd.DataFrame,
    metric: str = "auroc",
    output_path: Optional[str] = None,
    title: Optional[str] = None
) -> go.Figure:
    """
    Compare multiple models on a given metric.
    
    Args:
        results_df: DataFrame with columns ['model', 'train_metric', 'val_metric', 'test_metric']
        metric: Metric to compare
        output_path: Path to save HTML file (optional)
        title: Plot title (optional)
    
    Returns:
        Plotly figure object
    """
    if title is None:
        title = f"Model Comparison - {metric.upper()}"
    
    models = results_df['model'].values
    
    fig = go.Figure()
    
    for split, color in [('train', '#3498db'), ('val', '#e74c3c'), ('test', '#2ecc71')]:
        col_name = f'{split}_{metric}'
        if col_name in results_df.columns:
            fig.add_trace(go.Bar(
                name=split.capitalize(),
                x=models,
                y=results_df[col_name].values,
                marker_color=color,
                text=[f"{v:.3f}" for v in results_df[col_name].values],
                textposition='auto'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Model",
        yaxis_title=metric.upper(),
        template="plotly_white",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    if output_path:
        _save_figure(fig, output_path)
    
    return fig