import google.generativeai as genai
import os
import json

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sys
sys.stdout.reconfigure(encoding='utf-8')

genai.configure(api_key="AIzaSyD-a2kuCZWWzbiaQ9t34ELnJ4Y3HZT-Y3E")

vision_model = genai.GenerativeModel("gemini-1.5-flash")
text_model = genai.GenerativeModel("gemini-1.5-flash")

model = load_model(r"m32.h5")

class_labels = ['Cell', 'Cell-Multi', 'Cracking', 'Diode', 'Diode-Multi',
                'Hot-Spot', 'Hot-Spot-Multi', 'No-Anomaly', 'Offline-Module',
                'Shadowing', 'Soiling', 'Vegetation']

def predict_image(img: Image.Image):

    img = img.convert("RGB")
    img = img.resize((24, 40))  # (width, height)
    img_array = np.array(img)
    img_input = img_array.reshape(1, 40, 24, 3)

    prediction = model.predict(img_input)
    max_prob = np.max(prediction[0])
    class_ind = np.argmax(prediction[0])
    predicted_class = class_labels[class_ind]

    return predicted_class

def detect_anomalies(ir_image):
    prompt = """These are infrared  of a solar panel. Identify any faults such as hotspots, cracks, or soiling. Be specific about location and severity."""
    response = vision_model.generate_content([prompt, ir_image])
    return response.text

def check_warranty_eligibility(detected_issue, panel_metadata):
    prompt = f"""
    The issue detected is: {detected_issue}
    Panel installation date: {panel_metadata['install_date']}
    Warranty terms: {panel_metadata['warranty_terms']}
    Based on this, is the issue eligible for warranty?
    """
    response = text_model.generate_content(prompt)
    return response.text

def recommend_action(detected_issue):
    prompt = f"""The solar panel issue is: {detected_issue}
    Suggest repair steps, urgency level, and whether the panel should be replaced or cleaned."""
    response = text_model.generate_content(prompt)
    return response.text

def generate_customer_report(issue, warranty, actions):
    prompt = f"""Create a detailed report for the customer explaining but do not include the customer's name or contact information:
    - Detected issue: {issue}
    - Warranty status: {warranty}
    - Recommended next steps: {actions}
    Format the report for customer service use."""
    response = text_model.generate_content(prompt)
    return response.text

def solar_panel_diagnosis_pipeline(ir_img_path, metadata):

    ir_image = Image.open(ir_img_path)
    anomalies = predict_image(ir_image)
    warranty_status = check_warranty_eligibility(anomalies, metadata)
    recommendations = recommend_action(anomalies)

    report = generate_customer_report(anomalies, warranty_status, recommendations)

    return {
        "anomalies": anomalies,
        "warranty": warranty_status,
        "recommendations": recommendations,
        "report": report
    }

metadata = {
    "install_date": "2022-01-15",
    "warranty_terms": "25-year coverage for material and workmanship defects. Hot-spots covered within first 10 years."
}

results = solar_panel_diagnosis_pipeline(r"C:\Users\Harsh Chaudhary\Pictures\Camera Roll\IMG-20250509-WA0026.jpg" , metadata)
print(results["report"])
