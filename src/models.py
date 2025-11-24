"""
Model architectures for COVID-19 X-ray classification.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from . import config

class XrayClassifier(nn.Module):
    """
    Base classifier with transfer learning.
    """
    def __init__(self, model_name='resnet50', num_classes=config.NUM_CLASSES, pretrained=True):
        super(XrayClassifier, self).__init__()
        
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
            
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            
        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(config.DROPOUT_RATE / 2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def freeze_backbone(self):
        """Freeze backbone layers for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

def get_model(model_name='resnet50', num_classes=config.NUM_CLASSES, pretrained=True, freeze_backbone=True):
    """
    Get a model instance.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        freeze_backbone: Whether to freeze backbone layers
    
    Returns:
        model: PyTorch model
    """
    model = XrayClassifier(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    
    if freeze_backbone:
        model.freeze_backbone()
        print(f"Backbone frozen for transfer learning")
    
    return model

if __name__ == "__main__":
    # Test models
    print("Testing model architectures...")
    
    for model_name in ['resnet50', 'densenet121', 'efficientnet_b3']:
        print(f"\n{model_name}:")
        model = get_model(model_name=model_name, pretrained=False)
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
