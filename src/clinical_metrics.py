"""
Clinical performance metrics for COVID-19 X-ray classification.
Healthcare-specific evaluation metrics.
"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import pandas as pd
from . import config

def calculate_clinical_metrics(y_true, y_pred, y_proba=None, class_names=None):
    """
    Calculate comprehensive clinical metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (for ROC-AUC)
        class_names: List of class names
    
    Returns:
        dict: Dictionary of clinical metrics
    """
    if class_names is None:
        class_names = config.CLASSES
    
    metrics = {}
    
    # Overall metrics
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    metrics['classification_report'] = report
    
    # Calculate sensitivity, specificity, PPV, NPV for each class
    for i, class_name in enumerate(class_names):
        # Binary classification for this class vs all others
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
        
        # Sensitivity (Recall, True Positive Rate)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive Predictive Value (Precision)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        
        metrics[f'{class_name}_sensitivity'] = sensitivity
        metrics[f'{class_name}_specificity'] = specificity
        metrics[f'{class_name}_ppv'] = ppv
        metrics[f'{class_name}_npv'] = npv
        metrics[f'{class_name}_f1'] = f1
        metrics[f'{class_name}_tp'] = int(tp)
        metrics[f'{class_name}_tn'] = int(tn)
        metrics[f'{class_name}_fp'] = int(fp)
        metrics[f'{class_name}_fn'] = int(fn)
    
    # ROC-AUC if probabilities provided
    if y_proba is not None:
        # Multi-class ROC-AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['roc_auc_macro'] = roc_auc
            
            # Per-class ROC-AUC
            for i, class_name in enumerate(class_names):
                y_true_binary = (y_true == i).astype(int)
                y_proba_class = y_proba[:, i]
                class_roc_auc = roc_auc_score(y_true_binary, y_proba_class)
                metrics[f'{class_name}_roc_auc'] = class_roc_auc
        except:
            pass
    
    # Overall accuracy
    metrics['accuracy'] = (y_true == y_pred).mean()
    
    return metrics

def print_clinical_report(metrics, class_names=None):
    """
    Print a formatted clinical performance report.
    """
    if class_names is None:
        class_names = config.CLASSES
    
    print("=" * 80)
    print("CLINICAL PERFORMANCE REPORT")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    if 'roc_auc_macro' in metrics:
        print(f"Overall ROC-AUC (macro): {metrics['roc_auc_macro']:.4f}")
    
    print("\n" + "-" * 80)
    print("PER-CLASS CLINICAL METRICS")
    print("-" * 80)
    
    for class_name in class_names:
        print(f"\n{class_name}:")
        print(f"  Sensitivity (Recall): {metrics[f'{class_name}_sensitivity']:.4f}")
        print(f"  Specificity:          {metrics[f'{class_name}_specificity']:.4f}")
        print(f"  PPV (Precision):      {metrics[f'{class_name}_ppv']:.4f}")
        print(f"  NPV:                  {metrics[f'{class_name}_npv']:.4f}")
        print(f"  F1-Score:             {metrics[f'{class_name}_f1']:.4f}")
        if f'{class_name}_roc_auc' in metrics:
            print(f"  ROC-AUC:              {metrics[f'{class_name}_roc_auc']:.4f}")
        print(f"  TP: {metrics[f'{class_name}_tp']}, TN: {metrics[f'{class_name}_tn']}, "
              f"FP: {metrics[f'{class_name}_fp']}, FN: {metrics[f'{class_name}_fn']}")
    
    print("\n" + "=" * 80)
    
    # Highlight COVID-19 performance (most critical)
    if 'COVID' in class_names:
        print("\n🔴 COVID-19 DETECTION PERFORMANCE (CRITICAL):")
        print(f"   Sensitivity: {metrics['COVID_sensitivity']:.2%} (minimize false negatives!)")
        print(f"   Specificity: {metrics['COVID_specificity']:.2%}")
        print(f"   False Negatives: {metrics['COVID_fn']} (missed COVID cases)")
        print(f"   False Positives: {metrics['COVID_fp']} (unnecessary isolation)")
        print("=" * 80)

def save_metrics_to_csv(metrics, output_path, class_names=None):
    """
    Save metrics to CSV file.
    """
    if class_names is None:
        class_names = config.CLASSES
    
    # Create DataFrame for per-class metrics
    rows = []
    for class_name in class_names:
        row = {
            'Class': class_name,
            'Sensitivity': metrics[f'{class_name}_sensitivity'],
            'Specificity': metrics[f'{class_name}_specificity'],
            'PPV': metrics[f'{class_name}_ppv'],
            'NPV': metrics[f'{class_name}_npv'],
            'F1-Score': metrics[f'{class_name}_f1'],
            'TP': metrics[f'{class_name}_tp'],
            'TN': metrics[f'{class_name}_tn'],
            'FP': metrics[f'{class_name}_fp'],
            'FN': metrics[f'{class_name}_fn'],
        }
        if f'{class_name}_roc_auc' in metrics:
            row['ROC-AUC'] = metrics[f'{class_name}_roc_auc']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")

if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.rand(n_samples, n_classes)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
    
    metrics = calculate_clinical_metrics(y_true, y_pred, y_proba)
    print_clinical_report(metrics)
