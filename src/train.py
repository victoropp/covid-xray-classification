"""
Training script for COVID-19 X-ray classification models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path
import argparse

from . import config
from .models import get_model
from .data_loader import get_data_loaders
from .clinical_metrics import calculate_clinical_metrics, print_clinical_report, save_metrics_to_csv

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
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
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

def train_model(model_name='resnet50', num_epochs=config.NUM_EPOCHS, batch_size=config.BATCH_SIZE):
    """
    Train a model with early stopping and learning rate scheduling.
    """
    print("=" * 80)
    print(f"Training {model_name.upper()}")
    print("=" * 80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loaders
    print("\nLoading data...")
    # num_workers=0 to avoid Windows multiprocessing issues
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size, num_workers=0)
    
    # Calculate class weights
    class_weights_df = pd.read_csv(config.METADATA_DIR / "class_weights.csv")
    class_weights = [class_weights_df[cls].values[0] for cls in config.CLASSES]
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"\nClass weights: {class_weights}")
    
    # Model
    print(f"\nInitializing {model_name} model...")
    model = get_model(model_name=model_name, pretrained=True, freeze_backbone=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.REDUCE_LR_FACTOR, 
                                  patience=config.REDUCE_LR_PATIENCE)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = config.MODELS_DIR / f"{model_name}_best.pth"
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_labels, val_preds, val_probs = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_model_path)
            print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_labels, test_preds, test_probs = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Set Performance:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Calculate clinical metrics
    print("\nCalculating clinical metrics...")
    metrics = calculate_clinical_metrics(test_labels, test_preds, test_probs)
    print_clinical_report(metrics)
    
    # Save metrics
    metrics_path = config.MODELS_DIR / f"{model_name}_metrics.json"
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
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Save per-class metrics to CSV
    csv_path = config.MODELS_DIR / f"{model_name}_clinical_metrics.csv"
    save_metrics_to_csv(metrics, csv_path)
    
    # Save training history
    history_path = config.MODELS_DIR / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return model, history, metrics

def main():
    parser = argparse.ArgumentParser(description='Train COVID-19 X-ray classification model')
    parser.add_argument('--model', type=str, default='resnet50', 
                       choices=['resnet50', 'resnet101', 'densenet121', 'efficientnet_b3'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Train model
    model, history, metrics = train_model(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 80)
    print("All done! Model saved and ready for deployment.")
    print("=" * 80)

if __name__ == "__main__":
    main()
