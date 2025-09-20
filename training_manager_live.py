import streamlit as st
import os
import glob
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="YOLO Training Manager",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'models_data' not in st.session_state:
    st.session_state.models_data = []
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'selected_dataset_path' not in st.session_state:
    st.session_state.selected_dataset_path = "./yolov11/dataset.yaml"

class ModelManager:
    """Handles model discovery, analysis, and management"""
    
    def __init__(self, dataset_path=None):
        self.base_model_path = "./Model/yolo11x.pt"
        self.runs_dir = "./runs/detect"
        self.dataset_path = dataset_path or "./yolov11/dataset.yaml"
    
    def update_dataset_path(self, new_path):
        """Update the dataset path"""
        self.dataset_path = new_path
    
    def find_dataset_files(self):
        """Find all available dataset.yaml files in the project"""
        dataset_files = []
        
        # Common locations to search
        search_paths = [
            "./yolov11/dataset.yaml",
            "./Dataset/yolov11/dataset.yaml", 
            "./dataset.yaml",
            "./data/dataset.yaml",
            "./datasets/dataset.yaml"
        ]
        
        # Search in all subdirectories for dataset.yaml files
        for root, dirs, files in os.walk("."):
            for file in files:
                if file == "dataset.yaml":
                    full_path = os.path.join(root, file).replace("\\", "/")
                    if full_path not in search_paths:
                        search_paths.append(full_path)
        
        # Check which files actually exist
        for path in search_paths:
            if os.path.exists(path):
                dataset_files.append(path)
        
        return dataset_files
    
    def validate_dataset(self, dataset_path):
        """Validate if dataset file exists and is properly formatted"""
        if not os.path.exists(dataset_path):
            return False, f"Dataset file not found: {dataset_path}"
        
        try:
            import yaml
            with open(dataset_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['train', 'val', 'nc', 'names']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return False, f"Missing required fields: {missing_fields}"
            
            # Check if paths exist (relative to dataset file directory)
            dataset_dir = os.path.dirname(dataset_path)
            
            train_path = os.path.join(dataset_dir, data.get('train', ''))
            val_path = os.path.join(dataset_dir, data.get('val', ''))
            
            if not os.path.exists(train_path):
                return False, f"Training data not found: {train_path}"
            
            if not os.path.exists(val_path):
                return False, f"Validation data not found: {val_path}"
            
            return True, f"Valid dataset with {data['nc']} classes: {data['names']}"
            
        except Exception as e:
            return False, f"Error reading dataset file: {str(e)}"
    
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
                        f"Fine-tuned v{i+1} (Best)", 
                        f"{run_name}_best",
                        train_dir
                    )
                    models.append(model_info)
                
                # Check for last.pt
                last_path = os.path.join(weights_dir, "last.pt")
                if os.path.exists(last_path):
                    model_info = self._get_model_info(
                        last_path, 
                        f"Fine-tuned v{i+1} (Last)", 
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

def train_model_simple(model_path, config):
    """Simple training function that works like train.py"""
    try:
        # Load model
        model = YOLO(model_path)
        
        # Start training (simple, no complex threading)
        results = model.train(**config)
        
        return results, None
        
    except Exception as e:
        return None, str(e)

# Initialize components
model_manager = ModelManager(st.session_state.selected_dataset_path)

# Title and header
st.title("🎯 YOLO Training Manager")
st.markdown("### Manage, compare, and train YOLO models with ease")

# Sidebar for refresh and settings
with st.sidebar:
    st.header("🔧 Controls")
    
    if st.button("🔄 Refresh Models", key="refresh_models"):
        model_manager.update_dataset_path(st.session_state.selected_dataset_path)
        st.session_state.models_data = model_manager.scan_models()
        st.success("Models refreshed!")
    
    st.markdown("---")
    
    # Dataset Selection Section
    st.subheader("📊 Dataset Selection")
    
    # Find available datasets
    available_datasets = model_manager.find_dataset_files()
    
    if available_datasets:
        # Dataset selection dropdown
        selected_dataset = st.selectbox(
            "Choose Dataset",
            options=available_datasets,
            index=available_datasets.index(st.session_state.selected_dataset_path) 
                  if st.session_state.selected_dataset_path in available_datasets else 0,
            key="dataset_selector"
        )
        
        # Update session state if selection changed
        if selected_dataset != st.session_state.selected_dataset_path:
            st.session_state.selected_dataset_path = selected_dataset
            model_manager.update_dataset_path(selected_dataset)
            st.success(f"Dataset updated: {selected_dataset}")
            st.rerun()
        
        # Validate selected dataset
        is_valid, message = model_manager.validate_dataset(st.session_state.selected_dataset_path)
        
        if is_valid:
            st.success("✅ Valid Dataset")
            st.info(message)
        else:
            st.error("❌ Invalid Dataset")
            st.error(message)
        
        # Show dataset details
        with st.expander("📋 Dataset Details"):
            try:
                import yaml
                with open(st.session_state.selected_dataset_path, 'r') as f:
                    dataset_info = yaml.safe_load(f)
                
                st.write(f"**Classes ({dataset_info.get('nc', 'Unknown')}):**")
                for i, class_name in enumerate(dataset_info.get('names', [])):
                    st.write(f"  {i}: {class_name}")
                
                st.write(f"**Train Path:** {dataset_info.get('train', 'Not specified')}")
                st.write(f"**Val Path:** {dataset_info.get('val', 'Not specified')}")
                st.write(f"**Test Path:** {dataset_info.get('test', 'Not specified')}")
                
            except Exception as e:
                st.error(f"Error reading dataset: {e}")
    
    else:
        st.error("❌ No dataset.yaml files found")
        st.info("Expected locations:")
        st.text("• ./yolov11/dataset.yaml")
        st.text("• ./Dataset/yolov11/dataset.yaml")
        st.text("• ./dataset.yaml")
        
        # Manual path input
        st.subheader("� Manual Dataset Path")
        manual_path = st.text_input(
            "Enter dataset.yaml path:",
            value=st.session_state.selected_dataset_path,
            key="manual_dataset_path"
        )
        
        if st.button("✅ Use Manual Path", key="use_manual_path"):
            st.session_state.selected_dataset_path = manual_path
            model_manager.update_dataset_path(manual_path)
            st.success(f"Dataset path updated: {manual_path}")
            st.rerun()
    
    st.markdown("---")
    
    # Training configuration
    st.subheader("⚙️ Training Config")
    
    train_epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
    train_batch = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
    train_imgsz = st.selectbox("Image Size", [416, 480, 640, 800, 1024], index=2)
    train_device = st.selectbox("Device", ["0", "cpu", "0,1"], index=0)
    train_patience = st.number_input("Patience (Early Stopping)", min_value=10, max_value=100, value=30)

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
    
    # Create scrollable model list
    for idx, model in enumerate(st.session_state.models_data):
        with st.expander(f"🎯 {model['type']} - {model['modified_str']}", 
                        expanded=(st.session_state.selected_model == model['id'])):
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Path:** `{model['path']}`")
                st.write(f"**Size:** {model['size_mb']} MB")
                st.write(f"**ID:** {model['id']}")
            
            with col2:
                # Show metrics if available
                if model['metrics'] and 'error' not in model['metrics']:
                    st.write("**📊 Performance:**")
                    metrics = model['metrics']
                    if 'map50' in metrics:
                        st.write(f"mAP@50: {metrics['map50']:.3f}")
                    if 'final_loss' in metrics:
                        st.write(f"Loss: {metrics['final_loss']:.4f}")
                    if 'epochs' in metrics:
                        st.write(f"Epochs: {metrics['epochs']}")
            
            with col3:
                # Selection button
                if st.button(f"Select", key=f"select_{model['id']}", 
                           type="primary" if st.session_state.selected_model == model['id'] else "secondary"):
                    st.session_state.selected_model = model['id']
                    st.success(f"Selected!")
                    st.rerun()

# Training section
if st.session_state.selected_model:
    st.markdown("---")
    st.header("🚀 Training Controls")
    
    selected_model_data = next((m for m in st.session_state.models_data if m['id'] == st.session_state.selected_model), None)
    
    # Validate dataset before allowing training
    dataset_valid, dataset_message = model_manager.validate_dataset(st.session_state.selected_dataset_path)
    
    if selected_model_data and dataset_valid:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"**Selected Model:** {selected_model_data['type']}")
            st.code(f"Path: {selected_model_data['path']}")
            
            # Training config preview
            config_summary = f"""
            📋 Training Configuration:
            • Model: {selected_model_data['type']}
            • Dataset: {st.session_state.selected_dataset_path}
            • Epochs: {train_epochs}
            • Batch Size: {train_batch}
            • Image Size: {train_imgsz}
            • Device: {train_device}
            • Patience: {train_patience}
            """
            st.text(config_summary)
            
            # Dataset validation status
            st.success(f"✅ Dataset: {dataset_message}")
        
        with col2:
            if not st.session_state.training_in_progress:
                if st.button("🚀 Start Training", type="primary", key="start_training_btn"):
                    # Prepare training config
                    training_config = {
                        'data': st.session_state.selected_dataset_path,
                        'epochs': train_epochs,
                        'batch': train_batch,
                        'imgsz': train_imgsz,
                        'device': train_device,
                        'patience': train_patience,
                        'save': True,
                        'verbose': True,
                        'amp': True,
                        'workers': 0  # Set to 0 for Windows compatibility
                    }
                    
                    st.session_state.training_in_progress = True
                    st.success("🎓 Training started! Please wait...")
                    
                    # Show training info
                    with st.spinner("🔄 Training in progress..."):
                        # Simple training execution
                        results, error = train_model_simple(selected_model_data['path'], training_config)
                        
                        if error:
                            st.error(f"❌ Training failed: {error}")
                            st.session_state.training_in_progress = False
                        else:
                            st.success("✅ Training completed successfully!")
                            st.session_state.training_results = results
                            st.session_state.training_in_progress = False
                            
                            # Refresh models list
                            st.session_state.models_data = model_manager.scan_models()
                            st.info("🔄 Model list updated with new trained model")
                            
                            st.rerun()
            else:
                st.warning("🔄 Training in progress...")
                if st.button("⏹️ Stop Training", key="stop_training_btn"):
                    st.session_state.training_in_progress = False
                    st.warning("Training stop requested...")
                    st.rerun()
    
    elif selected_model_data and not dataset_valid:
        st.error("❌ Cannot start training - Dataset validation failed")
        st.error(dataset_message)
        st.info("Please select a valid dataset from the sidebar or fix the dataset configuration.")
    
    else:
        st.warning("⚠️ Model selection error - Please refresh and try again.")

# Training status display
if st.session_state.training_in_progress:
    st.markdown("---")
    st.header("� Training in Progress")
    st.info("Training is running... This may take a while depending on your configuration.")
    st.warning("Please do not close this window while training is in progress.")

# Training completion handling
if st.session_state.training_results and not st.session_state.training_in_progress:
    st.markdown("---")
    st.header("✅ Training Completed!")
    
    results = st.session_state.training_results
    
    if "error" in results:
        st.error(f"Training failed: {results['error']}")
    else:
        st.success("Training completed successfully!")
        
        # Show results
        if hasattr(results, 'save_dir'):
            st.info(f"📁 Results saved to: {results.save_dir}")
            
            # Check for new model files
            new_best = os.path.join(results.save_dir, "weights", "best.pt")
            new_last = os.path.join(results.save_dir, "weights", "last.pt")
            
            if os.path.exists(new_best):
                st.success(f"🏆 New best model: {new_best}")
            if os.path.exists(new_last):
                st.success(f"📄 Latest model: {new_last}")
        
        # Refresh models
        if st.button("🔄 Refresh Model List", key="refresh_after_training"):
            st.session_state.models_data = model_manager.scan_models()
            st.session_state.training_results = None
            st.success("Model list updated!")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("*💡 Tip: Training runs in the background. You can monitor progress in real-time!*")