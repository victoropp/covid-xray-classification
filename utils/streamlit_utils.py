"""
Streamlit utility functions for COVID-19 X-ray Classification app.
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def set_page_config(page_title="COVID-19 X-ray Classification"):
    """Set Streamlit page configuration."""
    st.set_page_config(
        page_title=page_title,
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def add_custom_css():
    """Add custom CSS styling."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .stAlert {
            margin-top: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

def display_metric_card(title, value, delta=None, help_text=None):
    """Display a metric card."""
    st.metric(label=title, value=value, delta=delta, help=help_text)

def create_confusion_matrix_plot(cm, class_names):
    """Create an interactive confusion matrix plot."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Class',
        yaxis_title='True Class',
        width=600,
        height=600
    )
    
    return fig

def create_roc_curve_plot(fpr, tpr, roc_auc, class_name):
    """Create ROC curve plot."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{class_name} (AUC = {roc_auc:.3f})',
        line=dict(width=2)
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {class_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate (Sensitivity)',
        width=600,
        height=600,
        showlegend=True
    )
    
    return fig

def create_metrics_comparison_plot(metrics_df):
    """Create bar plot comparing metrics across classes."""
    fig = go.Figure()
    
    metrics_to_plot = ['Sensitivity', 'Specificity', 'PPV', 'F1-Score']
    
    for metric in metrics_to_plot:
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Class'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto',
            ))
    
    fig.update_layout(
        title='Clinical Metrics Comparison',
        xaxis_title='Class',
        yaxis_title='Score',
        barmode='group',
        width=800,
        height=500,
        yaxis=dict(range=[0, 1])
    )
    
    return fig

def display_class_distribution(df, title="Class Distribution"):
    """Display class distribution pie chart."""
    class_counts = df['class'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=class_counts.index,
        values=class_counts.values,
        hole=0.3,
        textinfo='label+percent+value',
        textfont_size=14
    )])
    
    fig.update_layout(
        title=title,
        width=600,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def add_sidebar_info():
    """Add project information to sidebar."""
    with st.sidebar:
        st.markdown("### 🏥 COVID-19 X-ray Classification")
        st.markdown("**Medical AI for Clinical Decision Support**")
        st.markdown("---")
        st.markdown("**Author:** Victor Collins Oppon")
        st.markdown("**Project Type:** Healthcare AI")
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.markdown("- **Total Images:** 21,165")
        st.markdown("- **Classes:** 4")
        st.markdown("  - COVID-19: 3,616")
        st.markdown("  - Normal: 10,192")
        st.markdown("  - Lung Opacity: 6,012")
        st.markdown("  - Viral Pneumonia: 1,345")
        st.markdown("---")
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("- PyTorch & Torchvision")
        st.markdown("- Transfer Learning (ResNet, EfficientNet)")
        st.markdown("- Grad-CAM Explainability")
        st.markdown("- Streamlit Dashboard")

def display_warning_disclaimer():
    """Display medical disclaimer."""
    st.warning("""
    ⚠️ **Medical Disclaimer**: This is a research and educational tool for demonstrating AI capabilities. 
    It is NOT FDA-approved for clinical use and should NOT be used for actual medical diagnosis. 
    Always consult qualified healthcare professionals for medical decisions.
    """)
