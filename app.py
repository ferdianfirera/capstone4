import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image
import cv2
import numpy as np
import os
import re
import pandas as pd

# CONFIG
st.set_page_config(
    page_title="Food Calorie Detector",
    page_icon="🍛",
    layout="wide"
)

# Title
st.title("🍛 Food Calorie Detection")

with st.sidebar:
    #Upload Instruction
    st.header("Upload & Model Settings")
    st.markdown(
        """
        **How to use:**
        1. Select a model variant.
            - **RT-DETR**: Faster inference, good accuracy.
            - **YOLO11**: Balanced speed and accuracy.
            - **YOLO12m**: Higher accuracy, slower speed.
        2. Adjust the confidence threshold.
        3. Upload a food image (JPG, JPEG, PNG).
        4. The model will process and display detected food items.

        **Note:** Ensure the uploaded images are clear for better detection accuracy.
        """
    )
    st.markdown("---")

    option_model = st.selectbox(
        "Model Variant",
        options=["RT-DETR", "YOLO11", "YOLO12m"],
        index=0,
        key="model_variant"
    )
    #confidence threshold
    conf_slider = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        key="confidence_threshold"
    )    

    # IMAGE UPLOAD
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])

# LOAD MODEL
@st.cache_resource
def load_model():
    if option_model == "RT-DETR":
        model_path = "best_food_rtdetr.pt"
    elif option_model == "YOLO11":
        model_path = "best_food_yolo11.pt"
    else:
        model_path = "best_food_yolo12m.pt"
    return YOLO(model_path)

if 'last_model' not in st.session_state:
    st.session_state.last_model = option_model
elif st.session_state.last_model != option_model:
    st.session_state.last_model = option_model
    st.cache_resource.clear()
    st.rerun()  # trigger rerun

model = load_model()
st.success(f"✅ Model {option_model} loaded successfully!")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", width='stretch')
    
    # INFERENCE
    with st.spinner("Detecting objects..."):
        results = model.predict(image, conf=conf_slider, imgsz=640)
    
    # VISUALIZE RESULTS
    res_plotted = results[0].plot()
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    st.image(res_rgb, caption="🧩 Detection Results", width='stretch')
    
    # PARSE DETECTIONS
    boxes = results[0].boxes
    names = model.names
    total_calories = 0
    calories_list = []
    detected_foods = []

    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label = names[cls_id]
            conf = float(box.conf[0])
            calories = re.findall(r'\d+', names[cls_id])[0]
            food = label.split('-')[0].strip() 
            calories_list.append(int(calories))
            detected_foods.append((food, calories, conf))
        df = pd.DataFrame(detected_foods, columns=["Food", "Estimated Calories","Confidence Level"])
        st.subheader("🍱 Detected Food Items")
        st.table(df)    
        total_calories = sum(calories_list)
        print(total_calories)
        st.markdown(f"### 🔥 Total Estimated Calories: **{total_calories} kcal**")
    else:
        st.warning("No objects detected. Try a clearer image.")

st.markdown("---")
st.markdown(
    "Developed using **YOLO, RT-DETR** and **Streamlit** — Detect and estimate food calories easily!"
)
