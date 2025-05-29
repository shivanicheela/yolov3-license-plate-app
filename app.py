import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Load YOLOv8 model
model = YOLO('best.pt')

# Load PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("🚗 YOLOv8 License Plate Detection + PaddleOCR 🔍")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image to disk
    with open("input.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # Run YOLOv8 model
    results = model.predict(source="input.jpg", save=False)

    # Load original image
    image = cv2.imread("input.jpg")

    # Get boxes
    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        # Crop license plate region
        cropped_plate = image[y1:y2, x1:x2]

        # Run OCR on cropped image
        ocr_result = ocr.ocr(cropped_plate)

        # Get text
        text = ""
        if ocr_result and len(ocr_result[0]) > 0:
            text = ocr_result[0][0][1][0]  # extract the text only

        # Draw rectangle and put text
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (255, 0, 0), 2)

    # Convert BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    st.image(image_rgb, caption="Detected Plate with OCR", use_column_width=True)
