import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time

# Configure OpenCV to suppress MSMF warnings
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

# Page configuration
st.set_page_config(
    page_title="Real-time Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 Real-time Object Detection")
st.markdown("### Using your trained YOLOv11 model for object detection")

# Load model
@st.cache_resource
def load_model():
    model_path = "runs/detect/train/weights/best.pt"   # your model path
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.error(f"Model file not found at {model_path}")
        return None

model = load_model()

# ✅ Initialize camera state once
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False

if model is not None:
    # Sidebar
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05
    )
    detection_mode = st.sidebar.selectbox(
        "Select Detection Mode",
        ["Webcam", "Upload Image", "Upload Video"]
    )

    # =============== REALTIME WEBCAM ==================
    if detection_mode == "Webcam":
        st.header("📹 Live Camera Detection")

        col1, col2 = st.columns(2)
        with col1:
            start = st.button("Start Camera", type="primary")
        with col2:
            stop = st.button("Stop Camera")

        if start:
            st.session_state.run_camera = True
        if stop:
            st.session_state.run_camera = False

        frame_placeholder = st.empty()

        if st.session_state.run_camera:
            # Try different camera backends for better Windows compatibility
            cap = None
            for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
                cap = cv2.VideoCapture(0, backend)
                if cap.isOpened():
                    break
                cap.release()
            
            if not cap.isOpened():
                st.error("Cannot open camera. Please check your device and ensure no other application is using it.")
            else:
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                st.success("Camera started successfully!")
                
                frame_count = 0
                while st.session_state.run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Camera read failed. Retrying...")
                        time.sleep(0.1)
                        continue

                    # Process every 2nd frame for better performance
                    frame_count += 1
                    if frame_count % 2 == 0:
                        # Run YOLO detection
                        try:
                            results = model(frame, conf=confidence_threshold)
                            annotated = results[0].plot()
                            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                            # Show frame
                            frame_placeholder.image(annotated, channels="RGB", use_column_width=True)
                        except Exception as e:
                            st.error(f"Detection error: {e}")
                            continue

                    time.sleep(0.033)  # ~30 FPS
                    
                    # Check if stop was pressed by refreshing the session state
                    if not st.session_state.run_camera:
                        break
            
            cap.release()
            cv2.destroyAllWindows()

    # =============== IMAGE UPLOAD ==================
    elif detection_mode == "Upload Image":
        st.header("🖼️ Image Detection")
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)

            with col2:
                st.subheader("Detection Results")
                image_np = np.array(image)
                results = model(image_np, conf=confidence_threshold)
                annotated_image = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)

            # Detection summary
            if len(results[0].boxes) > 0:
                st.subheader("Detection Summary")
                detections = results[0].boxes
                class_counts = {}
                for box in detections:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    conf = float(box.conf[0])
                    if class_name not in class_counts:
                        class_counts[class_name] = []
                    class_counts[class_name].append(conf)

                for class_name, confs in class_counts.items():
                    avg_conf = np.mean(confs)
                    st.write(f"**{class_name}**: {len(confs)} detections (avg conf: {avg_conf:.2f})")
            else:
                st.info("No objects detected.")

    # =============== VIDEO UPLOAD ==================
    elif detection_mode == "Upload Video":
        st.header("🎬 Video Detection")
        uploaded_video = st.file_uploader(
            "Choose a video file...", 
            type=['mp4', 'avi', 'mov', 'mkv']
        )
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()

            if st.button("Process Video", type="primary"):
                cap = cv2.VideoCapture(tfile.name)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                st.info(f"Processing video: {total_frames} frames at {fps} FPS")

                progress = st.progress(0)
                frame_placeholder = st.empty()

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % 5 == 0:  # every 5th frame
                        results = model(frame, conf=confidence_threshold)
                        annotated = results[0].plot()
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, channels="RGB")
                    frame_count += 1
                    progress.progress(frame_count / total_frames)
                cap.release()
                st.success("Video processing completed!")

            try:
                os.unlink(tfile.name)
            except:
                pass

    # =============== SIDEBAR INFO ==================
    st.sidebar.header("Model Information")
    st.sidebar.info(f"**Classes**: {list(model.names.values())}")
    st.sidebar.info("**Model Path**: runs/detect/train/weights/best.pt")

else:
    st.error("Failed to load model.")
