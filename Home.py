"""
COVID-19 X-ray Classification - Home Page
Medical AI for Clinical Decision Support
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.streamlit_utils import set_page_config, add_custom_css, add_sidebar_info, display_warning_disclaimer

# Page config
set_page_config("COVID-19 X-ray Classification | Home")
add_custom_css()
add_sidebar_info()

# Main content
st.markdown('<h1 class="main-header">🏥 COVID-19 X-ray Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Medical AI for Clinical Decision Support</p>', unsafe_allow_html=True)

# Medical disclaimer
display_warning_disclaimer()

# Project overview
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Project Goal")
    st.markdown("""
    Develop a state-of-the-art deep learning system for classifying chest X-rays into four categories:
    - **COVID-19**
    - **Viral Pneumonia**
    - **Lung Opacity**
    - **Normal**
    
    This system serves as a **clinical decision support tool** to assist radiologists in rapid triage and diagnosis.
    """)

with col2:
    st.markdown("### 📊 Key Metrics")
    st.metric("Total Images", "21,165")
    st.metric("Model Accuracy", "68.7%", help="ResNet50 test set accuracy")
    st.metric("ROC-AUC (Macro)", "0.921", help="Excellent discrimination across classes")
    st.metric("Classes", "4")

with col3:
    st.markdown("### 🚀 Features")
    st.markdown("""
    - ✅ **Transfer Learning** (ResNet50, EfficientNet, DenseNet)
    - ✅ **Grad-CAM Explainability** (Visual interpretation)
    - ✅ **Clinical Metrics** (Sensitivity, Specificity, PPV, NPV)
    - ✅ **Real-time Predictions** (Interactive interface)
    - ✅ **Business Impact Analysis** (ROI calculator)
    """)

# Dataset overview
st.markdown("---")
st.markdown("## 📁 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Class Distribution")
    st.markdown("""
    | Class | Images | Percentage |
    |-------|--------|------------|
    | **Normal** | 10,192 | 48.2% |
    | **Lung Opacity** | 6,012 | 28.4% |
    | **COVID-19** | 3,616 | 17.1% |
    | **Viral Pneumonia** | 1,345 | 6.4% |
    
    **Note:** Class imbalance is handled through weighted loss and SMOTE augmentation.
    """)

with col2:
    st.markdown("### Data Split")
    st.markdown("""
    | Split | Images | Percentage |
    |-------|--------|------------|
    | **Training** | 14,815 | 70% |
    | **Validation** | 3,175 | 15% |
    | **Test** | 3,175 | 15% |
    
    **Stratified split** ensures balanced class distribution across all sets.
    """)

# Technical approach
st.markdown("---")
st.markdown("## 🛠️ Technical Approach")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Data Preprocessing")
    st.markdown("""
    - **CLAHE** for contrast enhancement
    - **Data Augmentation** (rotation, flip, brightness)
    - **Normalization** using ImageNet statistics
    - **Resize** to 224x224 for transfer learning
    """)

with col2:
    st.markdown("### 2️⃣ Model Training")
    st.markdown("""
    - **Transfer Learning** with ImageNet weights
    - **Class Weights** for imbalance handling
    - **Early Stopping** to prevent overfitting
    - **Learning Rate Scheduling** for optimization
    """)

with col3:
    st.markdown("### 3️⃣ Evaluation")
    st.markdown("""
    - **Clinical Metrics** (Sensitivity, Specificity)
    - **Confusion Matrix** analysis
    - **ROC-AUC** curves
    - **Grad-CAM** for explainability
    """)

# Business impact
st.markdown("---")
st.markdown("## 💼 Business Impact")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 🏥 Hospitals")
    st.markdown("""
    - Faster triage
    - Reduced radiologist workload
    - 24/7 availability
    """)

with col2:
    st.markdown("### 👨‍⚕️ Radiologists")
    st.markdown("""
    - AI-assisted diagnosis
    - Second opinion
    - Focus on complex cases
    """)

with col3:
    st.markdown("### 🌍 Public Health")
    st.markdown("""
    - Pandemic surveillance
    - Resource allocation
    - Early detection
    """)

with col4:
    st.markdown("### 💰 Cost Savings")
    st.markdown("""
    - Reduced diagnosis time
    - Optimized staffing
    - Better outcomes
    """)

# Navigation
st.markdown("---")
st.markdown("## 🧭 Navigation")

st.markdown("""
Use the sidebar to navigate through different sections:

1. **Executive Summary** - Model performance overview and key metrics
2. **Data Explorer** - Dataset visualization and statistics
3. **Model Performance** - Detailed model comparison and evaluation
4. **Clinical Predictions** - Interactive prediction interface with Grad-CAM
5. **Explainability** - Model interpretability and feature analysis
6. **Business Impact** - ROI calculator and cost-benefit analysis
7. **Industry Use Cases** - Real-world applications across healthcare sectors
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Author:</strong> Victor Collins Oppon | <strong>Project Type:</strong> Healthcare AI Portfolio Project</p>
    <p>Built with ❤️ using PyTorch, Streamlit, and Medical Imaging Best Practices</p>
</div>
""", unsafe_allow_html=True)
