"""
Data preprocessing and dataset creation for COVID-19 X-ray classification.
"""
import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from . import config

def create_metadata_csv():
    """
    Create metadata CSV files with image paths and labels.
    """
    print("Creating metadata CSV files...")
    
    all_data = []
    
    for class_name in config.CLASSES:
        class_dir = config.RAW_DATA_DIR / class_name / "images"
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} does not exist!")
            continue
        
        image_files = list(class_dir.glob("*.png"))
        print(f"Found {len(image_files)} images for class: {class_name}")
        
        for img_path in image_files:
            all_data.append({
                'image_path': str(img_path),
                'filename': img_path.name,
                'class': class_name,
                'class_idx': config.CLASSES.index(class_name)
            })
    
    df = pd.DataFrame(all_data)
    
    # Save full dataset metadata
    metadata_path = config.METADATA_DIR / "full_dataset.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Saved full dataset metadata: {metadata_path}")
    print(f"Total images: {len(df)}")
    print("\nClass distribution:")
    print(df['class'].value_counts())
    
    return df

def create_train_val_test_splits(df, random_state=config.RANDOM_SEED):
    """
    Create stratified train/val/test splits.
    """
    print("\nCreating train/val/test splits...")
    
    # First split: train + val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_RATIO,
        stratify=df['class'],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['class'],
        random_state=random_state
    )
    
    # Save splits
    train_df.to_csv(config.METADATA_DIR / "train.csv", index=False)
    val_df.to_csv(config.METADATA_DIR / "val.csv", index=False)
    test_df.to_csv(config.METADATA_DIR / "test.csv", index=False)
    
    print(f"Train set: {len(train_df)} images")
    print(f"Val set: {len(val_df)} images")
    print(f"Test set: {len(test_df)} images")
    
    print("\nTrain set class distribution:")
    print(train_df['class'].value_counts())
    print("\nVal set class distribution:")
    print(val_df['class'].value_counts())
    print("\nTest set class distribution:")
    print(test_df['class'].value_counts())
    
    return train_df, val_df, test_df

def calculate_class_weights(train_df):
    """
    Calculate class weights for handling imbalanced dataset.
    """
    class_counts = train_df['class'].value_counts()
    total_samples = len(train_df)
    
    class_weights = {}
    for class_name in config.CLASSES:
        count = class_counts.get(class_name, 1)
        weight = total_samples / (len(config.CLASSES) * count)
        class_weights[class_name] = weight
    
    print("\nClass weights:")
    for class_name, weight in class_weights.items():
        print(f"{class_name}: {weight:.4f}")
    
    # Save class weights
    weights_df = pd.DataFrame([class_weights])
    weights_df.to_csv(config.METADATA_DIR / "class_weights.csv", index=False)
    
    return class_weights

def main():
    """
    Main preprocessing pipeline.
    """
    print("=" * 60)
    print("COVID-19 X-ray Dataset Preprocessing")
    print("=" * 60)
    
    # Create metadata
    df = create_metadata_csv()
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_splits(df)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_df)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"Metadata saved to: {config.METADATA_DIR}")
    print(f"- full_dataset.csv: {len(df)} images")
    print(f"- train.csv: {len(train_df)} images")
    print(f"- val.csv: {len(val_df)} images")
    print(f"- test.csv: {len(test_df)} images")
    print(f"- class_weights.csv")

if __name__ == "__main__":
    main()
