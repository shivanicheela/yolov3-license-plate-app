import streamlit as st
import torch
from PIL import Image
import numpy as np
import tempfile
import cv2

# Load model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
    return model

model = load_model()

st.title("License Plate Detection (YOLO + best.pt)")
st.markdown("Upload an image to detect license plates.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy
    image_np = np.array(image)

    # Perform detection
    results = model(image_np)

    # Render results
    st.image(np.squeeze(results.render()), caption="Detected Image", use_column_width=True)
