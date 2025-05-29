import streamlit as st
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load model
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

st.title("License Plate Detection + OCR")

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    # Run YOLOv8 on image
    results = model.predict(source=image_path, save=False)

    for result in results:
        im = cv2.imread(image_path)
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            license_plate = im[y1:y2, x1:x2]

            # Run OCR
            if license_plate.size > 0:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as plate_file:
                    cv2.imwrite(plate_file.name, license_plate)
                    ocr_result = ocr.ocr(plate_file.name, cls=True)

                # Draw results
                if ocr_result:
                    for line in ocr_result:
                        for word in line:
                            _, text = word
                            st.write(f"📝 License Plate Text: {text}")

            # Draw bounding box
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show final image
        st.image(im, channels="BGR", caption="Detected Image")
