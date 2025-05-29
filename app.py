import streamlit as st
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

# Initialize PaddleOCR once
ocr = PaddleOCR(use_angle_cls=True, lang='en', gpu=False)

st.title("YOLOv3 License Plate Detection + OCR")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes from upload and convert to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", channels="BGR")

    # Load YOLO model
    model = YOLO('best.pt')  # Make sure best.pt is in the same folder or give full path

    # Run detection
    results = model.predict(source=img)

    # Get first result (detections for first image)
    det = results[0].boxes  # Boxes object

    # Copy image to draw boxes
    img_with_boxes = img.copy()

    detected_texts = []

    # Loop over detected boxes
    for box in det:
        # Box coordinates (xyxy)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw rectangle on image
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop detected plate region for OCR
        plate_img = img[y1:y2, x1:x2]

        # Convert BGR to RGB for PaddleOCR
        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)

        # Run OCR
        ocr_result = ocr.ocr(plate_img_rgb)

        # Extract text
        text = ""
        for line in ocr_result:
            text += line[1][0] + " "

        detected_texts.append(text.strip())

        # Put detected text near box
        cv2.putText(img_with_boxes, text.strip(), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show image with boxes and OCR text
    st.image(img_with_boxes, caption="Detected License Plates with OCR", channels="BGR")

    # Show all detected texts below image
    if detected_texts:
        st.write("Detected Text(s):")
        for idx, t in enumerate(detected_texts):
            st.write(f"{idx+1}: {t}")
    else:
        st.write("No license plates detected.")
