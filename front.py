import streamlit as st
from PIL import Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyD-a2kuCZWWzbiaQ9t34ELnJ4Y3HZT-Y3E")


# Your Gemini-powered functions (from previous answer)
from agent import (
    solar_panel_diagnosis_pipeline  # Import this from your core logic
)

st.set_page_config(page_title="Solar Panel Diagnosis", layout="wide")
st.title("🔍 Solar Panel Warranty Diagnosis Tool")

# Input Form
with st.form("diagnosis_form"):
    st.header("1. Upload Images")
    ir_img = st.file_uploader("Infrared Image", type=["jpg", "jpeg", "png"])

    st.header("2. Panel Metadata")
    install_date = st.date_input("Installation Date")
    warranty_terms = st.text_area("Warranty Terms", height=150, help="Paste or write warranty details here.")

    submitted = st.form_submit_button("Run Diagnosis")

if submitted and ir_img :
    st.success("Running Gemini-powered analysis...")

    metadata = {
        "install_date": str(install_date),
        "warranty_terms": warranty_terms,
    }

    results = solar_panel_diagnosis_pipeline(ir_img, metadata)

    st.subheader("📋 Full Report")
    st.markdown(results["report"])
    
    with st.expander("⚠️ Detected Anomalies"):
        st.markdown(results["anomalies"])
    
    with st.expander("✅ Warranty Status"):
        st.markdown(results["warranty"])

    with st.expander("🛠️ Recommendations"):
        st.markdown(results["recommendations"])
else:
    st.info("Please upload images and fill metadata to run diagnosis.")