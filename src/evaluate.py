"""
Evaluation script for trained COVID-19 X-ray classification models.
Loads a trained model and generates comprehensive metrics and visualizations.
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
import json
import argparse
from tqdm import tqdm

from . import config
from .models import get_model
from .data_loader import get_data_loaders
from .clinical_metrics import calculate_clinical_metrics, print_clinical_report, save_metrics_to_csv
from .gradcam import GradCAM, generate_gradcam_visualization

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        loss, accuracy, labels, predictions, probabilities
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(data_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASSES, 
                yticklabels=config.CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - COVID-19 X-ray Classification', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_roc_curves(y_true, y_probs, save_path):
    """Generate and save ROC curves for all classes."""
    n_classes = len(config.CLASSES)
    
    # Binarize labels for multi-class ROC
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{config.CLASSES[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class Classification', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to: {save_path}")
    
    return roc_auc

def plot_per_class_metrics(metrics, save_path):
    """Generate bar chart of per-class metrics."""
    classes = config.CLASSES
    metrics_to_plot = ['sensitivity', 'specificity', 'ppv', 'npv']
    metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        # Extract values using the flat key structure: {class_name}_{metric}
        values = [metrics[f'{cls}_{metric}'] for cls in classes]
        
        bars = axes[idx].bar(range(len(classes)), values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
        axes[idx].set_ylabel(name, fontsize=12)
        axes[idx].set_title(f'{name} by Class', fontsize=14, fontweight='bold')
        axes[idx].set_xticks(range(len(classes)))
        axes[idx].set_xticklabels(classes, rotation=45, ha='right')
        axes[idx].set_ylim([0, 1.05])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics chart saved to: {save_path}")

def generate_gradcam_examples(model, data_loader, device, save_dir, num_examples=12):
    """Generate Grad-CAM visualizations for sample images."""
    model.eval()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get samples from each class
    samples_per_class = num_examples // len(config.CLASSES)
    class_counts = {cls: 0 for cls in range(len(config.CLASSES))}
    
    print(f"\nGenerating Grad-CAM examples...")
    
    for images, labels in data_loader:
        for i in range(len(images)):
            label = labels[i].item()
            
            if class_counts[label] < samples_per_class:
                image_tensor = images[i:i+1].to(device)
                
                # Denormalize for visualization
                img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                mean = np.array(config.NORMALIZE_MEAN)
                std = np.array(config.NORMALIZE_STD)
                img_np = img_np * std + mean
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                # Generate Grad-CAM visualization
                save_path = save_dir / f"gradcam_{config.CLASSES[label]}_{class_counts[label]+1}.png"
                
                try:
                    overlayed, pred_class, confidence = generate_gradcam_visualization(
                        model,
                        image_tensor,
                        img_np,
                        class_names=config.CLASSES,
                        save_path=save_path
                    )
                    class_counts[label] += 1
                except Exception as e:
                    print(f"Warning: Failed to generate Grad-CAM for {config.CLASSES[label]}: {e}")
                    continue
        
        # Check if we have enough samples
        if all(count >= samples_per_class for count in class_counts.values()):
            break
    
    print(f"Generated {sum(class_counts.values())} Grad-CAM examples in: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained COVID-19 X-ray model')
    parser.add_argument('--model', type=str, default='resnet50',
                       choices=['resnet50', 'resnet101', 'densenet121', 'efficientnet_b3'],
                       help='Model architecture')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size for evaluation')
    parser.add_argument('--gradcam-examples', type=int, default=12,
                       help='Number of Grad-CAM examples to generate')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Evaluating {args.model.upper()} Model")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    model_path = config.MODELS_DIR / f"{args.model}_best.pth"
    if not model_path.exists():
        print(f"\nError: Model file not found at {model_path}")
        print("Please train the model first using: python -m src.train")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = get_model(model_name=args.model, pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("✓ Model loaded successfully")
    
    # Load data
    print("\nLoading data...")
    _, val_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=0)
    
    # Calculate class weights for loss
    class_weights_df = pd.read_csv(config.METADATA_DIR / "class_weights.csv")
    class_weights = [class_weights_df[cls].values[0] for cls in config.CLASSES]
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Evaluating on Test Set")
    print("=" * 80)
    
    test_loss, test_acc, test_labels, test_preds, test_probs = evaluate_model(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Set Performance:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Calculate clinical metrics
    print("\n" + "=" * 80)
    print("Clinical Metrics")
    print("=" * 80)
    
    metrics = calculate_clinical_metrics(test_labels, test_preds, test_probs)
    print_clinical_report(metrics)
    
    # Save metrics
    print("\n" + "=" * 80)
    print("Saving Metrics")
    print("=" * 80)
    
    metrics_path = config.MODELS_DIR / f"{args.model}_metrics.json"
    with open(metrics_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            elif isinstance(v, (np.int64, np.int32)):
                metrics_json[k] = int(v)
            elif isinstance(v, (np.float64, np.float32)):
                metrics_json[k] = float(v)
            elif isinstance(v, dict):
                metrics_json[k] = v
            else:
                metrics_json[k] = v
        json.dump(metrics_json, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Save per-class metrics to CSV
    csv_path = config.MODELS_DIR / f"{args.model}_clinical_metrics.csv"
    save_metrics_to_csv(metrics, csv_path)
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("Generating Visualizations")
    print("=" * 80)
    
    # Confusion matrix
    cm_path = config.REPORTS_DIR / f"{args.model}_confusion_matrix.png"
    plot_confusion_matrix(test_labels, test_preds, cm_path)
    
    # ROC curves
    roc_path = config.REPORTS_DIR / f"{args.model}_roc_curves.png"
    roc_auc_scores = plot_roc_curves(test_labels, test_probs, roc_path)
    
    # Per-class metrics
    metrics_chart_path = config.REPORTS_DIR / f"{args.model}_per_class_metrics.png"
    plot_per_class_metrics(metrics, metrics_chart_path)
    
    # Grad-CAM examples
    gradcam_dir = config.REPORTS_DIR / "gradcam_examples"
    generate_gradcam_examples(model, test_loader, device, gradcam_dir, args.gradcam_examples)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\nAll results saved to: {config.REPORTS_DIR}")
    print(f"  - Metrics: {metrics_path}")
    print(f"  - Confusion Matrix: {cm_path}")
    print(f"  - ROC Curves: {roc_path}")
    print(f"  - Per-class Metrics: {metrics_chart_path}")
    print(f"  - Grad-CAM Examples: {gradcam_dir}")

if __name__ == "__main__":
    main()
