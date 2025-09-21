import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Image Object Detection Test",
    page_icon="🎯",
    layout="wide"
)

# Title
st.title("🎯 Image Object Detection Test")
st.markdown("### Upload an image to test your trained YOLO model")

# Load model
@st.cache_resource
def load_model():
    model_path = "runs/segment/train6/weights/best.pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        st.success(f"✅ Model loaded successfully!")
        st.info(f"**Detected Classes**: {list(model.names.values())}")
        return model
    else:
        st.error(f"❌ Model file not found at {model_path}")
        return None

# Initialize model
model = load_model()

if model is not None:
    # Sidebar for settings
    st.sidebar.header("🔧 Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Lower values detect more objects but may include false positives"
    )
    
    # File uploader
    st.header("📁 Upload Image for Detection")
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF, WebP"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB"
        }
        
        with st.expander("📄 File Details"):
            for key, value in file_details.items():
                st.write(f"**{key}**: {value}")
        
        # Load and display the image
        try:
            image = Image.open(uploaded_file)
            
            # Create two columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(image, use_column_width=True)
                st.write(f"**Image size**: {image.size[0]} x {image.size[1]} pixels")
            
            with col2:
                st.subheader("🎯 Detection Results")
                
                # Add a button to run detection
                if st.button("🚀 Run Detection", type="primary"):
                    with st.spinner("🔍 Analyzing image..."):
                        # Convert PIL image to numpy array
                        image_np = np.array(image)
                        
                        # Run YOLO detection
                        results = model(image_np, conf=confidence_threshold)
                        
                        # Get the annotated image
                        annotated_image = results[0].plot()
                        
                        # Convert BGR to RGB for proper display
                        if len(annotated_image.shape) == 3:
                            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                        else:
                            annotated_image_rgb = annotated_image
                        
                        # Display the result
                        st.image(annotated_image_rgb, use_column_width=True)
                        
                        # Show detection statistics
                        detections = results[0].boxes
                        
                        if len(detections) > 0:
                            st.success(f"✅ Found {len(detections)} object(s)!")
                            
                            # Create detection summary
                            st.subheader("📊 Detection Summary")
                            
                            detection_data = []
                            for i, box in enumerate(detections):
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                confidence = float(box.conf[0])
                                
                                detection_data.append({
                                    "Object #": i + 1,
                                    "Class": class_name,
                                    "Confidence": f"{confidence:.2%}"
                                })
                            
                            # Display as a nice table
                            st.table(detection_data)
                            
                            # Show class distribution
                            class_counts = {}
                            for detection in detection_data:
                                class_name = detection["Class"]
                                if class_name in class_counts:
                                    class_counts[class_name] += 1
                                else:
                                    class_counts[class_name] = 1
                            
                            st.subheader("📈 Class Distribution")
                            for class_name, count in class_counts.items():
                                st.write(f"**{class_name}**: {count} detection(s)")
                                
                        else:
                            st.warning(f"⚠️ No objects detected with confidence ≥ {confidence_threshold:.1%}")
                            st.info("💡 Try lowering the confidence threshold if you expect objects in the image")
                
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
    
    else:
        # Show sample instructions when no file is uploaded
        st.info("👆 Please upload an image file to start detection")
        
        with st.expander("ℹ️ How to use this tool"):
            st.markdown("""
            1. **Upload an image** using the file uploader above
            2. **Adjust confidence threshold** in the sidebar if needed
            3. **Click 'Run Detection'** to analyze your image
            4. **View results** with bounding boxes and confidence scores
            
            **Tips:**
            - Higher confidence = fewer but more certain detections
            - Lower confidence = more detections but may include false positives
            - Supported formats: JPG, PNG, BMP, TIFF, WebP
            """)

else:
    st.error("❌ Cannot proceed without a trained model")
    st.info("Please ensure your model training is complete and the file exists at: `runs/detect/train/weights/best.pt`")