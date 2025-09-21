# app.py
import streamlit as st
import torch
import cv2
import tempfile
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Realtime Detection", layout="wide")

st.title("📷 YOLO Realtime Object Detection with Streamlit")

# Upload your YOLO model path
model_path = st.text_input("Enter YOLO model path (.pt file)", "runs/segment/train9/weights/best.pt")

# Load model only once
@st.cache_resource
def load_model(path):
    return YOLO(path)

if model_path:
    try:
        model = load_model(model_path)
        st.success(f"Model loaded from {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

run = st.checkbox("Run Camera")
FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)  # 0 = default webcam

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection
    results = model(frame, stream=True)

    # Draw results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"

            # Draw bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show frame
    FRAME_WINDOW.image(frame)

camera.release()