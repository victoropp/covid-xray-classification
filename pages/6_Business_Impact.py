"""
Business Impact - ROI Calculator & Cost-Benefit Analysis
"""
import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.streamlit_utils import set_page_config, add_custom_css, add_sidebar_info

set_page_config("Business Impact")
add_custom_css()
add_sidebar_info()

st.markdown('<h1 class="main-header">💼 Business Impact</h1>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("## 💰 ROI Calculator")

st.markdown("### Hospital Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    bed_count = st.number_input("Number of Beds", value=500, step=50)
    daily_xrays = st.number_input("Daily Chest X-rays", value=100, step=10)

with col2:
    radiologist_cost = st.number_input("Radiologist Cost ($/hour)", value=150, step=10)
    avg_read_time = st.number_input("Avg Read Time (minutes)", value=5, step=1)

with col3:
    covid_prevalence = st.slider("COVID Prevalence (%)", 0, 50, 10)
    implementation_cost = st.number_input("Implementation Cost ($)", value=100000, step=10000)

# Calculate ROI
st.markdown("---")
st.markdown("## 📊 Calculated Impact")

annual_xrays = daily_xrays * 365
time_saved_per_xray = avg_read_time * 0.5  # 50% time reduction
annual_time_saved = annual_xrays * time_saved_per_xray / 60  # hours
annual_cost_savings = annual_time_saved * radiologist_cost

# Additional benefits
covid_cases = annual_xrays * (covid_prevalence / 100)
faster_isolation = covid_cases * 1.5 * 150  # 1.5 hour faster isolation, $150/hour ED cost
reduced_transmission = covid_cases * 0.1 * 50000  # 10% reduction, $50K per case

total_annual_benefit = annual_cost_savings + faster_isolation + reduced_transmission
roi_percentage = ((total_annual_benefit - implementation_cost) / implementation_cost) * 100
payback_period = implementation_cost / (total_annual_benefit / 12)  # months

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Annual Cost Savings", f"${annual_cost_savings:,.0f}")
with col2:
    st.metric("Total Annual Benefit", f"${total_annual_benefit:,.0f}")
with col3:
    st.metric("ROI", f"{roi_percentage:.0f}%")
with col4:
    st.metric("Payback Period", f"{payback_period:.1f} months")

st.markdown("---")

st.markdown("## 📈 Benefit Breakdown")

st.markdown(f"""
**Direct Cost Savings:**
- Radiologist time saved: {annual_time_saved:,.0f} hours/year
- Cost savings: ${annual_cost_savings:,.0f}/year

**Clinical Benefits:**
- Faster COVID isolation: ${faster_isolation:,.0f}/year
- Reduced nosocomial transmission: ${reduced_transmission:,.0f}/year
- Improved patient outcomes: Priceless

**Total Annual Benefit:** ${total_annual_benefit:,.0f}

**Return on Investment:** {roi_percentage:.0f}% annually

**Payback Period:** {payback_period:.1f} months
""")

st.markdown("---")

st.markdown("## 🎯 Key Value Drivers")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ⏱️ Time Savings")
    st.markdown("""
    - 50-90% reduction in initial screening time
    - 24/7 availability without fatigue
    - Faster triage and isolation
    - Reduced ED wait times
    """)

with col2:
    st.markdown("### 💵 Cost Reduction")
    st.markdown("""
    - Reduced radiologist overtime
    - Optimized staffing levels
    - Fewer missed diagnoses
    - Lower malpractice risk
    """)

with col3:
    st.markdown("### 📊 Quality Improvement")
    st.markdown("""
    - Consistent performance
    - Reduced inter-observer variability
    - Better patient outcomes
    - Enhanced reputation
    """)
