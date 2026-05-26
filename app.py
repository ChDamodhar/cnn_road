import os
import subprocess
import json
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image

# -------------------------------------------------------------------------
# PAGE SETUP & HIGH-CONTRAST DARK THEME CSS
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Road Damage Detection", page_icon="🚧", layout="wide"
)

st.markdown(
    """
    <style>
    /* Force main app background */
    .stApp {
        background-color: #0B0F17 !important;
    }
    
    /* Text element visibility adjustments */
    .stMarkdown p, .stMarkdown li, p, li {
        color: #E2E8F0 !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    
    /* Header Typography styling */
    h1 {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 2.6rem !important;
        margin-bottom: 5px !important;
    }
    h2, h3 {
        color: #00E5FF !important; /* Bright Neon Cyan for Section Headers */
        font-weight: 700 !important;
        margin-top: 20px !important;
    }
    
    /* Context block layout */
    .about-box {
        background-color: #161D2A !important;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #243247;
        border-left: 6px solid #00E5FF;
        margin-bottom: 25px;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.5);
    }
    .about-box h4 {
        color: #00E5FF !important;
        font-size: 19px !important;
        font-weight: 700 !important;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
    }
    
    /* Statistical Summary UI panels */
    .metric-card {
        background-color: #161D2A !important;
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #243247;
        text-align: center;
        box-shadow: 0px 6px 14px rgba(0,0,0,0.4);
    }
    .metric-title {
        color: #94A3B8 !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        font-weight: 600 !important;
        display: block;
        margin-bottom: 10px;
    }
    
    /* Operations/Urgency contextual banners */
    .rec-box-high {
        background-color: #3B1414 !important;
        padding: 22px;
        border-radius: 10px;
        border: 1px solid #632020;
        border-left: 6px solid #FF3B30;
        margin-top: 15px;
    }
    .rec-box-medium {
        background-color: #33200A !important;
        padding: 22px;
        border-radius: 10px;
        border: 1px solid #543510;
        border-left: 6px solid #FF9500;
        margin-top: 15px;
    }
    .rec-box-low {
        background-color: #0F291B !important;
        padding: 22px;
        border-radius: 10px;
        border: 1px solid #1A472E;
        border-left: 6px solid #34C759;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------------
# SECTION 1 — Header Layout
# -------------------------------------------------------------------------
st.title("AI-Based Road Damage Detection System")
st.markdown(
    "<p style='color: #00E5FF !important; font-size: 19px !important; font-weight: 600; margin-top:-10px; margin-bottom: 25px;'>Smart City Infrastructure Monitoring using CNN</p>", 
    unsafe_allow_html=True
)
st.markdown("<hr style='border: 1px solid #243247;'/>", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# SECTION 2 — Technical Context
# -------------------------------------------------------------------------
with st.expander("ℹ️ About the Project & Industry Context", expanded=True):
    st.markdown(
        """
        <div class="about-box">
            <h4>Why Road Monitoring is Important</h4>
            <p>Unmonitored road damage like deep potholes and severe structural cracks significantly 
            increases vehicle maintenance overhead, compromises commuter safety, and slows logistics metrics. 
            Automating tracking prevents fatal highway failures and grid gridlocks.</p>
            
            <h4>Role of CNN in Computer Vision</h4>
            <p>Convolutional Neural Networks (CNNs) emulate mammalian visual structures by running spatial filter 
            convolutions over matrix arrays. They capture edge configurations, textures, and shape features 
            independently of shifting scales, allowing automated extraction of infrastructure anomalies from simple image feeds.</p>
            
            <h4>Practical Industry Applications</h4>
            <ul>
                <li><b style="color: #FFFFFF;">Smart City Municipal Dashboards:</b> Directing real-time automated dispatch logs directly to localized maintenance teams.</li>
                <li><b style="color: #FFFFFF;">Autonomous Vehicles (AV):</b> Assisting visual navigation arrays to slow down dynamically before structural anomalies.</li>
                <li><b style="color: #FFFFFF;">Highway Fleet Logistics:</b> Dynamic rerouting algorithms avoiding heavily degraded freight corridors.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Establish side-by-side dashboard partition columns
col_left, col_right = st.columns([1, 1.2], gap="large")

# -------------------------------------------------------------------------
# SECTION 3 — Image Ingestion Interface
# -------------------------------------------------------------------------
with col_left:
    st.markdown("<h2>📸 Data Ingestion</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload structural highway imagery for diagnostic analysis",
        type=["jpg", "jpeg", "png"],
        help="Supports Drag-and-Drop or direct exploratory browsing.",
    )

    # -------------------------------------------------------------------------
    # SECTION 4 — File Rendering
    # -------------------------------------------------------------------------
    if uploaded_file is not None:
        st.markdown("<h3>Preview Uploaded Road Image</h3>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Source Feed Image")

        temp_img_path = f"temp_{uploaded_file.name}"
        image.save(temp_img_path)

# -------------------------------------------------------------------------
# MODEL PIPELINE SUBPROCESS RUNNER
# -------------------------------------------------------------------------
if uploaded_file is not None:
    with col_right:
        st.markdown("<h2>🎯 Machine Learning Diagnostics</h2>", unsafe_allow_html=True)

        with st.spinner("Processing image matrix through CNN stack..."):
            try:
                worker_script = "predict_worker.py"
                
                # Fetch absolute path of the environment's python runner
                active_python_env = sys.executable

                # Execute pipeline using explicitly isolated environment context
                result = subprocess.run(
                    [active_python_env, worker_script, temp_img_path],
                    capture_output=True,
                    text=True,
                    check=False  
                )

                # Route internal exceptions directly to screen codeblocks
                if result.returncode != 0:
                    st.error("❌ **CNN Prediction Pipeline Error Details:**")
                    st.code(result.stderr, language="bash")
                    st.stop()

                raw_predictions = json.loads(result.stdout.strip())
                class_labels = ["pothole", "crack", "manhole"]

                top_idx = np.argmax(raw_predictions)
                predicted_class = class_labels[top_idx].title()
                confidence = raw_predictions[top_idx] * 100

                # Structural vulnerability logic maps
                if predicted_class == "Pothole":
                    severity = "High" if confidence > 75 else "Medium"
                elif predicted_class == "Crack":
                    severity = "Medium" if confidence > 60 else "Low"
                else:
                    severity = "Low"

                # -------------------------------------------------------------
                # SECTION 5 — Inference Metrics Visualization
                # -------------------------------------------------------------
                st.markdown("<h3>Inference Metrics Summary</h3>", unsafe_allow_html=True)
                m_col1, m_col2, m_col3 = st.columns(3)

                with m_col1:
                    st.markdown(
                        f"""<div class="metric-card">
                            <span class="metric-title">Prediction</span>
                            <b style="color:#FF3B30; font-size:24px; font-weight:800;">{predicted_class}</b>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m_col2:
                    st.markdown(
                        f"""<div class="metric-card">
                            <span class="metric-title">Confidence</span>
                            <b style="color:#FFCC00; font-size:24px; font-weight:800;">{confidence:.1f}%</b>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                with m_col3:
                    color_map = {"High": "#FF3B30", "Medium": "#FF9500", "Low": "#34C759"}
                    st.markdown(
                        f"""<div class="metric-card">
                            <span class="metric-title">Severity Level</span>
                            <b style="color:{color_map[severity]}; font-size:24px; font-weight:800;">{severity}</b>
                        </div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("<br>", unsafe_allow_html=True)

                # -------------------------------------------------------------
                # SECTION 6 — Plotly Class Probability Bar Chart
                # -------------------------------------------------------------
                st.markdown("<h3>Class Confidence Graph</h3>", unsafe_allow_html=True)
                df_chart = pd.DataFrame(
                    {
                        "Damage Type": [c.title() for c in class_labels],
                        "Probability (%)": [p * 100 for p in raw_predictions],
                    }
                )

                fig = px.bar(
                    df_chart,
                    x="Probability (%)",
                    y="Damage Type",
                    orientation="h",
                    text="Probability (%)",
                    color="Probability (%)",
                    color_continuous_scale="Reds",
                    range_x=[0, 100],
                )

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#FFFFFF", size=13, family="sans-serif"),
                    height=240,
                    margin=dict(l=20, r=20, t=10, b=10),
                    showlegend=False,
                    xaxis=dict(showgrid=True, gridcolor="#243247", title_font=dict(color="#94A3B8")),
                    yaxis=dict(title_font=dict(color="#94A3B8"))
                )
                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(color="#FFFFFF", size=12)
                )
                st.plotly_chart(fig, use_container_width=True)

                # -------------------------------------------------------------
                # SECTION 7 — Civil Engineering Advisory Warnings
                # -------------------------------------------------------------
                st.markdown("<h3>Operational Recommendations</h3>", unsafe_allow_html=True)
                if severity == "High":
                    st.markdown(
                        """
                        <div class="rec-box-high">
                            <b style="font-size:18px; color:#FFD3D3; display:block; margin-bottom:5px;">🚨 Immediate maintenance recommended.</b>
                            <span style="color:#FFFFFF; font-size:15px;"><b>Safety Warning:</b> High-risk road condition detected. Structural voiding may trigger active suspension compression issues or severe dynamic swerving behaviors.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                elif severity == "Medium":
                    st.markdown(
                        """
                        <div class="rec-box-medium">
                            <b style="font-size:18px; color:#FFE6CC; display:block; margin-bottom:5px;">📅 Scheduled maintenance recommended.</b>
                            <span style="color:#FFFFFF; font-size:15px;"><b>Safety Warning:</b> Moderate road structural distress. Surface requires routine resurfacing patch work within the current deployment cycle to prevent full pavement failure.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="rec-box-low">
                            <b style="font-size:18px; color:#D4EDDA; display:block; margin-bottom:5px;">✅ Routine diagnostic monitoring recommended.</b>
                            <span style="color:#FFFFFF; font-size:15px;"><b>Safety Warning:</b> Standard road surface environment or expected utility layout. Risk index is normal. No structural emergency repairs required.</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            except Exception as ex:
                st.error(f"Inference pipeline execution encountered an error: {str(ex)}")
            finally:
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
else:
    with col_right:
        st.info("💡 Awaiting image upload from data ingestion panel to begin CNN evaluation loop.")