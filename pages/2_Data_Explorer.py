"""
Data Explorer - Dataset Visualization and Statistics
"""
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.streamlit_utils import set_page_config, add_custom_css, add_sidebar_info, display_class_distribution

set_page_config("Data Explorer")
add_custom_css()
add_sidebar_info()

st.markdown('<h1 class="main-header">📁 Data Explorer</h1>', unsafe_allow_html=True)

st.markdown("---")

# Load metadata
try:
    metadata_dir = project_root / "data" / "metadata"
    full_df = pd.read_csv(metadata_dir / "full_dataset.csv")
    train_df = pd.read_csv(metadata_dir / "train.csv")
    val_df = pd.read_csv(metadata_dir / "val.csv")
    test_df = pd.read_csv(metadata_dir / "test.csv")
    
    # Dataset Overview
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", f"{len(full_df):,}")
    with col2:
        st.metric("Training Set", f"{len(train_df):,}")
    with col3:
        st.metric("Validation Set", f"{len(val_df):,}")
    with col4:
        st.metric("Test Set", f"{len(test_df):,}")
    
    st.markdown("---")
    
    # Class Distribution
    st.markdown("## 🎯 Class Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Full Dataset")
        display_class_distribution(full_df, "Full Dataset Distribution")
    
    with col2:
        st.markdown("### Training Set")
        display_class_distribution(train_df, "Training Set Distribution")
    
    # Detailed Statistics
    st.markdown("---")
    st.markdown("## 📈 Detailed Statistics")
    
    # Create summary table
    summary_data = []
    for class_name in full_df['class'].unique():
        full_count = len(full_df[full_df['class'] == class_name])
        train_count = len(train_df[train_df['class'] == class_name])
        val_count = len(val_df[val_df['class'] == class_name])
        test_count = len(test_df[test_df['class'] == class_name])
        
        summary_data.append({
            'Class': class_name,
            'Total': full_count,
            'Train': train_count,
            'Val': val_count,
            'Test': test_count,
            'Percentage': f"{100*full_count/len(full_df):.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Class Imbalance Analysis
    st.markdown("---")
    st.markdown("## ⚖️ Class Imbalance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Imbalance Ratio")
        class_counts = full_df['class'].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1", 
                 help="Ratio of largest to smallest class")
        
        st.markdown(f"""
        - **Largest Class:** {class_counts.index[0]} ({class_counts.values[0]:,} images)
        - **Smallest Class:** {class_counts.index[-1]} ({class_counts.values[-1]:,} images)
        - **Imbalance Factor:** {imbalance_ratio:.2f}x
        """)
    
    with col2:
        st.markdown("### Mitigation Strategies")
        st.markdown("""
        **Applied Techniques:**
        1. ✅ **Class Weights** in loss function
        2. ✅ **Stratified Splitting** for balanced sets
        3. ✅ **Data Augmentation** for minority classes
        4. ✅ **Evaluation Metrics** (sensitivity, not just accuracy)
        
        **Result:** Model performs well across all classes despite imbalance.
        """)
    
    # Data Quality
    st.markdown("---")
    st.markdown("## ✅ Data Quality")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Completeness")
        st.markdown("""
        - ✅ No missing images
        - ✅ All labels verified
        - ✅ Metadata available
        - ✅ Consistent format (PNG)
        """)
    
    with col2:
        st.markdown("### Preprocessing")
        st.markdown("""
        - ✅ CLAHE contrast enhancement
        - ✅ Resized to 224x224
        - ✅ RGB conversion
        - ✅ ImageNet normalization
        """)
    
    with col3:
        st.markdown("### Augmentation")
        st.markdown("""
        - ✅ Horizontal flip
        - ✅ Rotation (±10°)
        - ✅ Brightness/Contrast
        - ✅ Gaussian noise
        """)

except Exception as e:
    st.error(f"Error loading dataset metadata: {e}")
    st.info("Please run preprocessing first: `python -m src.preprocessing`")

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Dataset: COVID-19 Radiography Database</p>
    <p>21,165 chest X-ray images across 4 classes</p>
</div>
""", unsafe_allow_html=True)
