"""
PyTorch Dataset and DataLoader for COVID-19 X-ray images.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
from PIL import Image
# import albumentations as A  # Removed due to compatibility issues
# from albumentations.pytorch import ToTensorV2  # Removed - not needed
from . import config

class XrayDataset(Dataset):
    """
    Custom Dataset for X-ray images.
    """
    def __init__(self, dataframe, transform=None, augment=False):
        """
        Args:
            dataframe: pandas DataFrame with columns ['image_path', 'class', 'class_idx']
            transform: torchvision transforms
            augment: whether to apply data augmentation
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.augment = augment
        
        # Albumentations augmentation pipeline (for training only)
        if self.augment:
            # Simple torchvision augmentations for training
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomResizedCrop(size=config.IMG_SIZE, scale=(0.9, 1.0))
            ])
        else:
            self.aug_transform = None
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.df.loc[idx, 'image_path']
        label = self.df.loc[idx, 'class_idx']
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply CLAHE for better contrast
        image = self.apply_clahe(image)
        
        # Resize to target size (numpy)
        image = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        
        # Convert to PIL Image for torchvision transforms
        image = Image.fromarray(image)
        
        # Apply augmentation if training
        if self.aug_transform is not None:
            image = self.aug_transform(image)
        
        # Apply transforms (normalization, etc.)
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def apply_clahe(self, image):
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        Improves contrast in X-ray images.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image

def get_transforms(augment=False):
    """
    Get image transforms for training or validation/test.
    """
    if augment:
        # Training transforms with augmentation and normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    else:
        # Validation/Test transforms (only normalization)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.NORMALIZE_MEAN,
                std=config.NORMALIZE_STD
            )
        ])
    
    return transform

def get_data_loaders(batch_size=config.BATCH_SIZE, num_workers=4):
    """
    Create DataLoaders for train, val, and test sets.
    """
    # Load metadata
    train_df = pd.read_csv(config.METADATA_DIR / "train.csv")
    val_df = pd.read_csv(config.METADATA_DIR / "val.csv")
    test_df = pd.read_csv(config.METADATA_DIR / "test.csv")
    
    # Create datasets
    train_dataset = XrayDataset(
        train_df,
        transform=get_transforms(augment=False),
        augment=True  # Augmentation handled by albumentations
    )
    
    val_dataset = XrayDataset(
        val_df,
        transform=get_transforms(augment=False),
        augment=False
    )
    
    test_dataset = XrayDataset(
        test_df,
        transform=get_transforms(augment=False),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test data loading
    print("Testing data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=8, num_workers=0)
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    print("Data loaders working correctly!")
