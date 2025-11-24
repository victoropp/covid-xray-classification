"""
Model Performance - Detailed Model Comparison
"""
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.streamlit_utils import set_page_config, add_custom_css, add_sidebar_info

set_page_config("Model Performance")
add_custom_css()
add_sidebar_info()

st.markdown('<h1 class="main-header">📊 Model Performance</h1>', unsafe_allow_html=True)

st.markdown("---")

# Load metrics
models_dir = project_root / "models"
model_name = "resnet50"
metrics_path = models_dir / f"{model_name}_metrics.json"
clinical_metrics_path = models_dir / f"{model_name}_clinical_metrics.csv"
history_path = models_dir / f"{model_name}_history.json"

if metrics_path.exists() and clinical_metrics_path.exists():
    # Load data
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    clinical_df = pd.read_csv(clinical_metrics_path)
    
    # Overview Metrics
    st.markdown(f"## 🏆 {model_name.upper()} Performance Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    with col2:
        macro_f1 = metrics['classification_report']['macro avg']['f1-score']
        st.metric("Macro F1-Score", f"{macro_f1:.1%}")
    with col3:
        weighted_recall = metrics['classification_report']['weighted avg']['recall']
        st.metric("Weighted Recall", f"{weighted_recall:.1%}")
    with col4:
        st.metric("Macro ROC-AUC", f"{metrics['roc_auc_macro']:.3f}")
    
    st.markdown("---")
    
    # Clinical Metrics Table
    st.markdown("## 🏥 Clinical Metrics by Class")
    st.markdown("Detailed breakdown of performance for each condition.")
    
    # Format the dataframe for display
    display_df = clinical_df.copy()
    for col in display_df.columns:
        if col != 'Class':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
            
    st.dataframe(display_df, use_container_width=True)
    
    st.markdown("""
    **Metric Definitions:**
    - **Sensitivity (Recall):** Ability to correctly identify positive cases (Critical for COVID-19).
    - **Specificity:** Ability to correctly identify negative cases.
    - **PPV (Precision):** Probability that a positive prediction is actually positive.
    - **NPV:** Probability that a negative prediction is actually negative.
    """)
    
    st.markdown("---")
    
    # Training History
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        st.markdown("## 📈 Training History")
        
        # Create dataframe for plotting
        epochs = range(1, len(history['train_loss']) + 1)
        hist_df = pd.DataFrame({
            'Epoch': epochs,
            'Train Loss': history['train_loss'],
            'Val Loss': history['val_loss'],
            'Train Acc': history['train_acc'],
            'Val Acc': history['val_acc']
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_loss = px.line(hist_df, x='Epoch', y=['Train Loss', 'Val Loss'], 
                              title='Loss over Epochs',
                              labels={'value': 'Loss', 'variable': 'Metric'})
            st.plotly_chart(fig_loss, use_container_width=True)
            
        with col2:
            fig_acc = px.line(hist_df, x='Epoch', y=['Train Acc', 'Val Acc'], 
                             title='Accuracy over Epochs',
                             labels={'value': 'Accuracy', 'variable': 'Metric'})
            st.plotly_chart(fig_acc, use_container_width=True)

else:
    st.warning("Model metrics not found. Please train the model first.")
    st.info("""
    To train the model, run:
    ```bash
    python -m src.train --model resnet50 --epochs 30 --batch-size 32
    ```
    """)
    
    # Show placeholder/expected performance
    st.markdown("## 📈 Expected Performance (Placeholder)")
    st.markdown("""
    Based on similar medical imaging tasks, we expect:

    | Model | Accuracy | COVID Sensitivity | Parameters | Training Time |
    |-------|----------|-------------------|------------|---------------|
    | ResNet50 | 95%+ | 96%+ | 25.6M | ~2 hours |
    | EfficientNet-B3 | 94%+ | 95%+ | 12.2M | ~3 hours |
    | DenseNet121 | 94%+ | 95%+ | 8.0M | ~2.5 hours |
    """)
