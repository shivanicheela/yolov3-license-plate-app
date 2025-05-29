import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.title("🚗 License Plate Detection (YOLOv8)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_path = "temp.jpg"
    image.save(image_path)

    model = YOLO("best.pt")

    with st.spinner("Detecting..."):
        results = model.predict(source=image_path, save=True)
    
    st.success("Done!")

    # Show prediction result
    result_img_path = os.path.join(results[0].save_dir, "temp.jpg")
    st.image(result_img_path, caption="Detected License Plate", use_column_width=True)
