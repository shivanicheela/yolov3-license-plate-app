import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import cv2

# Load YOLOv8 model and PaddleOCR
model = YOLO("best.pt")  # Make sure best.pt is in the same repo
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("🚘 License Plate Detection and OCR")

# Upload an image
uploaded_file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 detection
    results = model(img_array)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Crop detected license plate region
            plate_crop = img_array[y1:y2, x1:x2]

            # Run OCR on the cropped region
            ocr_result = ocr.ocr(plate_crop, cls=True)

            if ocr_result and len(ocr_result[0]) > 0:
                text = ocr_result[0][0][1][0]
                st.write("🔤 Detected License Plate Text:", text)
            else:
                st.write("❌ No text detected on license plate")

            # Draw bounding box on image
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show final image with boxes
    st.image(img_array, caption="Detected Plate(s)", use_column_width=True)
