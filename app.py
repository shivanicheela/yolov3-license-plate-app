import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load the model
model = YOLO("best3.pt")

st.title("YOLOv3 License Plate Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv3 model
    with st.spinner("Detecting license plate..."):
        results = model.predict(source=image, save=False, conf=0.25)
        boxes = results[0].boxes

    # Draw and show results
    results[0].show()
    st.image(results[0].plot(), caption="Detection Result", use_column_width=True)
