"""
Explainability - Model Interpretability & Feature Analysis
"""
import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.streamlit_utils import set_page_config, add_custom_css, add_sidebar_info

set_page_config("Explainability")
add_custom_css()
add_sidebar_info()

st.markdown('<h1 class="main-header">🔍 Explainability</h1>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 🎯 Why Explainability Matters in Healthcare")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Clinical Trust")
    st.markdown("""
    - Radiologists need to understand AI decisions
    - Validates that model focuses on relevant features
    - Identifies potential failure modes
    - Builds confidence in AI assistance
    """)

with col2:
    st.markdown("### Regulatory Requirements")
    st.markdown("""
    - FDA requires explainability for medical AI
    - Transparency for patient safety
    - Accountability for clinical decisions
    - Audit trail for quality assurance
    """)

st.markdown("---")

st.markdown("## 🔬 Grad-CAM: Visual Interpretation")

st.info("""
**Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizes which regions of the X-ray the model focuses on when making predictions.

**How it works:**
1. Forward pass through the network
2. Compute gradients of target class with respect to feature maps
3. Weight feature maps by gradient importance
4. Create heatmap showing important regions
5. Overlay heatmap on original image
""")

st.markdown("### Example Grad-CAM Visualizations")

st.markdown("""
**COVID-19 Case:**
- Model focuses on bilateral ground-glass opacities
- Highlights peripheral lung regions
- Consistent with radiological findings

**Normal Case:**
- Model shows diffuse attention across lung fields
- No focal areas of concern
- Validates absence of pathology

**Viral Pneumonia:**
- Focal consolidation in lower lobes
- Different pattern from COVID-19
- Helps distinguish between conditions
""")

st.markdown("---")

st.markdown("## 📊 Model Decision Process")

st.markdown("""
### Step-by-Step Interpretation

1. **Image Preprocessing**
   - CLAHE contrast enhancement
   - Resize to 224×224
   - Normalize with ImageNet statistics

2. **Feature Extraction**
   - Pre-trained ResNet50 backbone
   - Hierarchical feature learning
   - Low-level (edges) → High-level (patterns)

3. **Classification**
   - Custom classifier head
   - Dropout for regularization
   - Softmax for probability distribution

4. **Grad-CAM Generation**
   - Identify important feature maps
   - Weight by gradient importance
   - Create visual heatmap

5. **Clinical Interpretation**
   - Overlay heatmap on X-ray
   - Validate anatomical focus
   - Provide confidence score
""")

st.markdown("---")

st.markdown("## ✅ Validation of Model Behavior")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### What We Want to See")
    st.markdown("""
    ✅ Focus on lung regions (not artifacts)
    ✅ Bilateral attention for COVID-19
    ✅ Focal attention for pneumonia
    ✅ Diffuse attention for normal cases
    ✅ Consistent with radiological findings
    """)

with col2:
    st.markdown("### Red Flags")
    st.markdown("""
    ❌ Focus on image borders or markers
    ❌ Attention to patient labels or text
    ❌ Inconsistent patterns across similar cases
    ❌ Contradicts clinical knowledge
    ❌ High confidence with poor localization
    """)

st.markdown("---")

st.markdown("## 🔮 Future Enhancements")

st.markdown("""
- **SHAP Values**: Feature importance analysis
- **Attention Maps**: Multi-head attention visualization
- **Counterfactual Explanations**: "What if" scenarios
- **Uncertainty Quantification**: Confidence intervals
- **Comparative Analysis**: Side-by-side case comparison
""")
