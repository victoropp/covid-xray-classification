"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for model explainability.
Visualizes which regions of the X-ray the model focuses on for predictions.
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from . import config

class GradCAM:
    """
    Grad-CAM implementation for CNNs.
    """
    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for Grad-CAM (e.g., 'layer4' for ResNet)
        """
        self.model = model
        self.model.eval()
        
        # Get the target layer
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, target_layer):
                self.target_layer = getattr(model.backbone, target_layer)
            else:
                # For models like EfficientNet
                self.target_layer = model.backbone.features[-1]
        else:
            self.target_layer = getattr(model, target_layer)
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save activations."""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            cam: Grad-CAM heatmap (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, target_class]
        class_score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy()
    
    def visualize(self, image, cam, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) or (C, H, W) tensor
            cam: Grad-CAM heatmap (H, W)
            alpha: Overlay transparency
            colormap: OpenCV colormap
        
        Returns:
            overlayed_image: Image with heatmap overlay (H, W, 3)
        """
        # Convert image to numpy if tensor
        if torch.is_tensor(image):
            # Denormalize
            image = image.cpu().numpy()
            if image.shape[0] == 3:  # (C, H, W)
                image = image.transpose(1, 2, 0)
            
            # Denormalize using ImageNet stats
            mean = np.array(config.NORMALIZE_MEAN)
            std = np.array(config.NORMALIZE_STD)
            image = image * std + mean
            image = np.clip(image, 0, 1)
        
        # Ensure image is uint8
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Resize CAM to image size
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Apply colormap
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed = (1 - alpha) * image + alpha * heatmap
        overlayed = overlayed.astype(np.uint8)
        
        return overlayed, heatmap

def generate_gradcam_visualization(model, image_tensor, original_image, class_names=None, save_path=None):
    """
    Generate and save Grad-CAM visualization.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        original_image: Original image for visualization
        class_names: List of class names
        save_path: Path to save visualization
    
    Returns:
        overlayed_image: Image with Grad-CAM overlay
        predicted_class: Predicted class name
        confidence: Prediction confidence
    """
    if class_names is None:
        class_names = config.CLASSES
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted_idx = probs.max(dim=1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()
    
    # Generate Grad-CAM
    gradcam = GradCAM(model, target_layer='layer4')  # Adjust for different models
    cam = gradcam.generate_cam(image_tensor, target_class=predicted_idx.item())
    
    # Visualize
    overlayed_image, heatmap = gradcam.visualize(image_tensor[0], cam, alpha=config.GRADCAM_ALPHA)
    
    # Create figure with subplots
    if save_path:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original X-ray', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap)
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlayed_image)
        axes[2].set_title(f'Prediction: {predicted_class}\nConfidence: {confidence:.2%}', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Grad-CAM visualization saved to: {save_path}")
    
    return overlayed_image, predicted_class, confidence

if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("Use generate_gradcam_visualization() to create explainability visualizations.")
