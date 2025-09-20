import streamlit as st
import os
import glob
import json
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import subprocess
import threading
import time
import queue
import re

# Page configuration
st.set_page_config(
    page_title="YOLO Training Manager",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'models_data' not in st.session_state:
    st.session_state.models_data = []

class ModelManager:
    """Handles model discovery, analysis, and management"""
    
    def __init__(self):
        self.base_model_path = "./Model/yolo11x.pt"
        self.runs_dir = "./runs/detect"
        self.dataset_path = "./yolov11/dataset.yaml"
    
    def scan_models(self):
        """Scan and catalog all available models"""
        models = []
        
        # Add base model
        if os.path.exists(self.base_model_path):
            base_info = self._get_model_info(self.base_model_path, "Base Model", "yolo11x")
            models.append(base_info)
        
        # Scan training runs
        if os.path.exists(self.runs_dir):
            train_dirs = sorted(glob.glob(os.path.join(self.runs_dir, "train*")))
            
            for i, train_dir in enumerate(train_dirs):
                run_name = os.path.basename(train_dir)
                weights_dir = os.path.join(train_dir, "weights")
                
                # Check for best.pt
                best_path = os.path.join(weights_dir, "best.pt")
                if os.path.exists(best_path):
                    model_info = self._get_model_info(
                        best_path, 
                        f"Fine-tuned v{i+1}", 
                        f"{run_name}_best",
                        train_dir
                    )
                    models.append(model_info)
                
                # Check for last.pt
                last_path = os.path.join(weights_dir, "last.pt")
                if os.path.exists(last_path):
                    model_info = self._get_model_info(
                        last_path, 
                        f"Latest v{i+1}", 
                        f"{run_name}_last",
                        train_dir
                    )
                    models.append(model_info)
        
        return models
    
    def _get_model_info(self, model_path, model_type, model_id, train_dir=None):
        """Extract detailed information about a model"""
        
        # Basic file info
        stat = os.stat(model_path)
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        model_info = {
            'id': model_id,
            'path': model_path,
            'type': model_type,
            'size_mb': round(size_mb, 2),
            'modified': modified,
            'modified_str': modified.strftime("%Y-%m-%d %H:%M:%S"),
            'train_dir': train_dir,
            'metrics': {},
            'args': {}
        }
        
        # Try to load model metrics and training info
        if train_dir:
            # Load training results
            results_path = os.path.join(train_dir, "results.csv")
            if os.path.exists(results_path):
                try:
                    df = pd.read_csv(results_path)
                    if not df.empty:
                        last_row = df.iloc[-1]
                        model_info['metrics'] = {
                            'epochs': len(df),
                            'final_loss': round(float(last_row.get('train/box_loss', 0)), 4),
                            'map50': round(float(last_row.get('metrics/mAP50(B)', 0)), 4),
                            'map50_95': round(float(last_row.get('metrics/mAP50-95(B)', 0)), 4),
                            'precision': round(float(last_row.get('metrics/precision(B)', 0)), 4),
                            'recall': round(float(last_row.get('metrics/recall(B)', 0)), 4)
                        }
                except Exception as e:
                    model_info['metrics']['error'] = str(e)
            
            # Load training args
            args_path = os.path.join(train_dir, "args.yaml")
            if os.path.exists(args_path):
                try:
                    import yaml
                    with open(args_path, 'r') as f:
                        args_data = yaml.safe_load(f)
                        model_info['args'] = {
                            'epochs': args_data.get('epochs', 'N/A'),
                            'batch_size': args_data.get('batch', 'N/A'),
                            'img_size': args_data.get('imgsz', 'N/A'),
                            'device': args_data.get('device', 'N/A')
                        }
                except Exception as e:
                    model_info['args']['error'] = str(e)
        
        return model_info
    
    def get_next_run_name(self):
        """Get the next available run name"""
        if not os.path.exists(self.runs_dir):
            return "train"
        
        existing_runs = glob.glob(os.path.join(self.runs_dir, "train*"))
        if not existing_runs:
            return "train"
        
        # Find highest number
        max_num = 0
        for run_path in existing_runs:
            run_name = os.path.basename(run_path)
            if run_name == "train":
                max_num = max(max_num, 1)
            elif run_name.startswith("train"):
                try:
                    num = int(run_name[5:])  # Remove "train" prefix
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        return f"train{max_num + 1}" if max_num > 0 else "train2"

# Initialize model manager
model_manager = ModelManager()

# Title and header
st.title("🎯 YOLO Training Manager")
st.markdown("### Manage, compare, and train YOLO models with ease")

# Sidebar for refresh and settings
with st.sidebar:
    st.header("🔧 Controls")
    
    if st.button("🔄 Refresh Models", key="refresh_models"):
        st.session_state.models_data = model_manager.scan_models()
        st.success("Models refreshed!")
    
    st.markdown("---")
    
    # Training configuration
    st.subheader("⚙️ Training Config")
    
    train_epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)
    train_batch = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
    train_imgsz = st.selectbox("Image Size", [416, 480, 640, 800, 1024], index=2)
    train_device = st.selectbox("Device", ["0", "cpu", "0,1"], index=0)
    
    st.markdown("---")
    
    # Dataset info
    st.subheader("📊 Dataset Info")
    if os.path.exists(model_manager.dataset_path):
        st.success("✅ Dataset found")
        st.text(f"Path: {model_manager.dataset_path}")
    else:
        st.error("❌ Dataset not found")
        st.text(f"Expected: {model_manager.dataset_path}")

# Load models data
if not st.session_state.models_data:
    with st.spinner("🔍 Scanning for models..."):
        st.session_state.models_data = model_manager.scan_models()

# Main content area
if not st.session_state.models_data:
    st.warning("⚠️ No models found!")
    st.info("Please ensure you have:")
    st.markdown("- Base model at `./Model/yolo11x.pt`")
    st.markdown("- Or completed training runs in `./runs/detect/`")
else:
    # Models overview
    st.header("📚 Available Models")
    
    # Create model cards
    cols = st.columns(min(3, len(st.session_state.models_data)))
    
    for idx, model in enumerate(st.session_state.models_data):
        col_idx = idx % len(cols)
        
        with cols[col_idx]:
            # Model card
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin: 10px 0;">
                    <h4>🎯 {model['type']}</h4>
                    <p><strong>ID:</strong> {model['id']}</p>
                    <p><strong>Size:</strong> {model['size_mb']} MB</p>
                    <p><strong>Modified:</strong> {model['modified_str']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics display
                if model['metrics']:
                    with st.expander("📊 Metrics"):
                        for key, value in model['metrics'].items():
                            if key != 'error':
                                st.metric(key.replace('_', ' ').title(), value)
                
                # Selection button
                button_key = f"select_{model['id']}"
                if st.button(f"Select {model['type']}", key=button_key, type="primary" if st.session_state.selected_model == model['id'] else "secondary"):
                    st.session_state.selected_model = model['id']
                    st.success(f"Selected: {model['type']}")

# Selected model details and training
if st.session_state.selected_model:
    st.markdown("---")
    
    # Find selected model
    selected_model_data = next((m for m in st.session_state.models_data if m['id'] == st.session_state.selected_model), None)
    
    if selected_model_data:
        st.header(f"🎯 Selected: {selected_model_data['type']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Model details
            st.subheader("📋 Model Details")
            
            details_df = pd.DataFrame([
                {"Property": "Path", "Value": selected_model_data['path']},
                {"Property": "Type", "Value": selected_model_data['type']},
                {"Property": "Size", "Value": f"{selected_model_data['size_mb']} MB"},
                {"Property": "Modified", "Value": selected_model_data['modified_str']},
            ])
            
            st.dataframe(details_df, hide_index=True, use_container_width=True)
            
            # Training arguments if available
            if selected_model_data['args']:
                st.subheader("⚙️ Previous Training Config")
                args_df = pd.DataFrame([
                    {"Parameter": k.replace('_', ' ').title(), "Value": v} 
                    for k, v in selected_model_data['args'].items()
                    if k != 'error'
                ])
                if not args_df.empty:
                    st.dataframe(args_df, hide_index=True, use_container_width=True)
        
        with col2:
            # Training metrics if available
            if selected_model_data['metrics'] and 'error' not in selected_model_data['metrics']:
                st.subheader("📈 Performance Metrics")
                
                metrics = selected_model_data['metrics']
                
                if 'map50' in metrics:
                    st.metric("mAP@50", f"{metrics['map50']:.3f}")
                if 'map50_95' in metrics:
                    st.metric("mAP@50-95", f"{metrics['map50_95']:.3f}")
                if 'precision' in metrics:
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                if 'recall' in metrics:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                if 'final_loss' in metrics:
                    st.metric("Final Loss", f"{metrics['final_loss']:.4f}")
                if 'epochs' in metrics:
                    st.metric("Epochs Trained", metrics['epochs'])

# Training section
st.markdown("---")
st.header("🚀 Start Training")

if not st.session_state.selected_model:
    st.info("👆 Please select a model above to start training")
elif not os.path.exists(model_manager.dataset_path):
    st.error("❌ Dataset not found. Please ensure your dataset.yaml exists.")
else:
    selected_model_data = next((m for m in st.session_state.models_data if m['id'] == st.session_state.selected_model), None)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"🎯 **Ready to train from:** {selected_model_data['type']}")
        st.info(f"📁 **Model path:** {selected_model_data['path']}")
        
        # Training configuration summary
        st.subheader("📋 Training Configuration")
        config_df = pd.DataFrame([
            {"Parameter": "Starting Model", "Value": selected_model_data['type']},
            {"Parameter": "Epochs", "Value": train_epochs},
            {"Parameter": "Batch Size", "Value": train_batch},
            {"Parameter": "Image Size", "Value": train_imgsz},
            {"Parameter": "Device", "Value": train_device},
            {"Parameter": "Dataset", "Value": model_manager.dataset_path},
        ])
        st.dataframe(config_df, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("🎮 Training Controls")
        
        if not st.session_state.training_in_progress:
            if st.button("🚀 Start Training", type="primary", key="start_training"):
                st.session_state.training_in_progress = True
                st.session_state.training_logs = []
                st.success("🎓 Training started!")
                st.rerun()
        else:
            st.warning("🔄 Training in progress...")
            if st.button("⏹️ Stop Training", type="secondary", key="stop_training"):
                st.session_state.training_in_progress = False
                st.info("Training stopped by user")
                st.rerun()

# Training progress section
if st.session_state.training_in_progress:
    st.markdown("---")
    st.header("📊 Training Progress")
    
    # This would be expanded with actual training implementation
    progress_placeholder = st.empty()
    logs_placeholder = st.empty()
    
    with progress_placeholder.container():
        st.info("🔄 Training simulation - In actual implementation, this would show real-time progress")
        
        # Placeholder for actual training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate training progress (replace with actual implementation)
        for i in range(100):
            progress_bar.progress((i + 1) / 100)
            status_text.text(f"Epoch {i+1}/{train_epochs} - Simulated training...")
            time.sleep(0.1)
            
            if not st.session_state.training_in_progress:
                break
        
        if st.session_state.training_in_progress:
            st.success("✅ Training completed!")
            st.session_state.training_in_progress = False
            
            # Refresh models list
            st.session_state.models_data = model_manager.scan_models()
            st.info("🔄 Model list updated with new trained model")
            
            time.sleep(2)
            st.rerun()

# Footer
st.markdown("---")
st.markdown("*💡 Tip: Use the refresh button to update the model list after training*")