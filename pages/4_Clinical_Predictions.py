"""
Clinical Predictions - Interactive Prediction Interface
"""
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.streamlit_utils import set_page_config, add_custom_css, add_sidebar_info, display_warning_disclaimer
from src import config
from src.models import get_model
from src.gradcam import GradCAM

set_page_config("Clinical Predictions")
add_custom_css()
add_sidebar_info()

st.markdown('<h1 class="main-header">🔬 Clinical Predictions</h1>', unsafe_allow_html=True)

display_warning_disclaimer()

st.markdown("---")

# Load model
@st.cache_resource
def load_trained_model(model_name='resnet50'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = project_root / "models" / f"{model_name}_best.pth"
    
    if not model_path.exists():
        return None, None
    
    # Initialize model
    model = get_model(model_name=model_name, pretrained=False)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, device

model, device = load_trained_model()

if model is None:
    st.warning("Model not found. Please train the model first.")
    st.info("Run `python -m src.train --model resnet50` to train.")
else:
    st.success(f"Model loaded successfully! Device: {device}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Upload X-ray")
        uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded X-ray', use_container_width=True)
            
            # Preprocess for model
            img_array = np.array(image)
            
            # Apply CLAHE (same as training)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # Resize
            img_resized = cv2.resize(img_processed, (config.IMG_SIZE, config.IMG_SIZE))
            
            # Normalize and convert to tensor
            img_tensor = torch.from_numpy(img_resized).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1) # HWC -> CHW
            
            # Normalize with ImageNet stats
            mean = torch.tensor(config.NORMALIZE_MEAN).view(3, 1, 1)
            std = torch.tensor(config.NORMALIZE_STD).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dim

            # Prediction
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                conf, pred_idx = torch.max(probs, 0)
                pred_class = config.CLASSES[pred_idx.item()]
                pred_label = config.CLASS_NAMES[pred_class]

    with col2:
        if uploaded_file is not None:
            st.markdown("### Analysis Results")
            
            # Display prediction
            color = "red" if pred_class == "COVID" else "green" if pred_class == "Normal" else "orange"
            st.markdown(f"""
            <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid {color};'>
                <h2 style='margin:0; color: {color};'>{pred_label}</h2>
                <p style='margin:0; font-size: 1.2em;'>Confidence: <strong>{conf.item():.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Probability Distribution")
            
            # Create probability chart
            probs_dict = {config.CLASS_NAMES[c]: p.item() for c, p in zip(config.CLASSES, probs)}
            sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
            
            for label, prob in sorted_probs.items():
                st.progress(prob)
                st.text(f"{label}: {prob:.1%}")
            
            # Grad-CAM
            st.markdown("#### Explainability (Grad-CAM)")
            with st.spinner("Generating heatmap..."):
                gradcam = GradCAM(model, target_layer='layer4')
                cam = gradcam.generate_cam(img_tensor, target_class=pred_idx.item())
                
                # Visualize overlay
                overlayed_img, heatmap_img = gradcam.visualize(img_tensor[0], cam, alpha=0.4)
                st.image(overlayed_img, caption=f"Grad-CAM Attention Map ({pred_label})", use_container_width=True)
                
            st.info("Heatmap shows regions contributing most to the prediction (Red = High Attention).")
