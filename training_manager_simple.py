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
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'models_data' not in st.session_state:
    st.session_state.models_data = []
if 'selected_dataset_path' not in st.session_state:
    st.session_state.selected_dataset_path = "./yolov11/dataset.yaml"
if 'current_directory' not in st.session_state:
    st.session_state.current_directory = os.path.abspath(".")
if 'browse_mode' not in st.session_state:
    st.session_state.browse_mode = False

class FileBrowser:
    """Simple file browser for selecting dataset files"""
    
    @staticmethod
    def list_directory(path):
        """List contents of a directory"""
        try:
            if not os.path.exists(path):
                return [], []
            
            dirs = []
            files = []
            
            for item in sorted(os.listdir(path)):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    # Skip hidden directories and common non-useful ones
                    if not item.startswith('.') and item not in ['__pycache__', 'node_modules', '.git']:
                        dirs.append(item)
                elif item.endswith(('.yaml', '.yml')):
                    files.append(item)
            
            return dirs, files
        except (PermissionError, OSError):
            return [], []
    
    @staticmethod
    def get_parent_directory(path):
        """Get parent directory"""
        parent = os.path.dirname(path)
        # Avoid going above the drive root on Windows
        if len(parent) < 3 and ':' in parent:
            return path
        return parent

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
            'metrics': {}
        }
        
        # Try to load model metrics if available
        if train_dir:
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
                        }
                except Exception:
                    pass
        
        return model_info

def train_model(model_path, config):
    """Simple training function that works like train.py"""
    try:
        model = YOLO(model_path)
        results = model.train(**config)
        return results, None
    except Exception as e:
        return None, str(e)

# Initialize components
model_manager = ModelManager(st.session_state.selected_dataset_path)

# Title and header
st.title("🎯 YOLO Training Manager")
st.markdown("### Simple and effective model training interface")

# Sidebar for dataset selection and training config
with st.sidebar:
    st.header("📊 Dataset Selection")
    
    # Dataset selection mode
    selection_mode = st.radio(
        "Selection Mode:",
        ["📋 Quick Select", "📁 Browse Files"],
        key="dataset_mode"
    )
    
    if selection_mode == "📋 Quick Select":
        # Find available datasets
        available_datasets = model_manager.find_dataset_files()
        
        if available_datasets:
            # Dataset selection dropdown
            selected_dataset = st.selectbox(
                "Choose Dataset",
                options=available_datasets,
                index=available_datasets.index(st.session_state.selected_dataset_path) 
                      if st.session_state.selected_dataset_path in available_datasets else 0,
            )
            
            # Update session state if selection changed
            if selected_dataset != st.session_state.selected_dataset_path:
                st.session_state.selected_dataset_path = selected_dataset
                model_manager.update_dataset_path(selected_dataset)
                st.success(f"Dataset updated!")
                st.rerun()
        else:
            st.warning("❌ No dataset.yaml files found automatically")
    
    elif selection_mode == "📁 Browse Files":
        # File browser interface
        st.subheader("📂 File Browser")
        
        # Current directory display
        current_dir = st.session_state.current_directory
        st.text(f"📍 Current: {current_dir}")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬆️ Parent", key="go_parent"):
                parent_dir = FileBrowser.get_parent_directory(current_dir)
                if parent_dir != current_dir:  # Not at root
                    st.session_state.current_directory = parent_dir
                    st.rerun()
        
        with col2:
            if st.button("🏠 Project Root", key="go_home"):
                st.session_state.current_directory = os.path.abspath(".")
                st.rerun()
        
        with col3:
            # Jump to Dataset folder if it exists
            dataset_dir = os.path.abspath("Dataset")
            if os.path.exists(dataset_dir):
                if st.button("📦 Dataset Folder", key="go_dataset"):
                    st.session_state.current_directory = dataset_dir
                    st.rerun()
        
        # Quick shortcuts to common dataset locations
        st.write("🔗 **Quick Shortcuts:**")
        shortcut_cols = st.columns(4)
        
        # Common dataset paths
        common_paths = [
            ("Dataset", "Dataset"),
            ("yolov11", "yolov11"),
            ("data", "data"),
            ("datasets", "datasets")
        ]
        
        for i, (name, path) in enumerate(common_paths):
            full_path = os.path.abspath(path)
            if os.path.exists(full_path):
                with shortcut_cols[i]:
                    if st.button(f"📁 {name}", key=f"shortcut_{name}"):
                        st.session_state.current_directory = full_path
                        st.rerun()
        
        # Directory contents
        dirs, files = FileBrowser.list_directory(current_dir)
        
        # Show directories
        if dirs:
            st.write("📁 **Directories:**")
            dir_cols = st.columns(min(3, len(dirs)))  # Max 3 columns
            for i, directory in enumerate(dirs[:15]):  # Limit to 15 for UI
                col_idx = i % 3
                with dir_cols[col_idx]:
                    if st.button(f"📁 {directory}", key=f"dir_{i}_{directory}", use_container_width=True):
                        new_path = os.path.join(current_dir, directory)
                        st.session_state.current_directory = new_path
                        st.rerun()
        
        # Show YAML files
        if files:
            st.write("📄 **YAML Files:**")
            for i, file in enumerate(files):
                file_path = os.path.join(current_dir, file)
                # Display file with select button
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file}")
                with col2:
                    if st.button("Select", key=f"select_file_{i}_{file}"):
                        normalized_path = file_path.replace("\\", "/")
                        st.session_state.selected_dataset_path = normalized_path
                        model_manager.update_dataset_path(normalized_path)
                        st.success(f"✅ Selected: {file}")
                        st.balloons()
                        st.rerun()
        else:
            st.info("ℹ️ No YAML files in this directory")
            
        # Debug info (can be removed later)
        with st.expander("🔍 Debug Info"):
            st.write(f"Current directory: {current_dir}")
            st.write(f"Directories found: {len(dirs)}")
            st.write(f"YAML files found: {len(files)}")
            st.write(f"All files in dir: {os.listdir(current_dir) if os.path.exists(current_dir) else 'Directory not accessible'}")
        
        # Manual path input as fallback
        st.markdown("---")
        st.write("**💡 Or enter path manually:**")
        manual_path = st.text_input(
            "Full path to dataset.yaml:",
            value=st.session_state.selected_dataset_path,
            key="manual_path_input"
        )
        if st.button("✅ Use This Path", key="use_manual"):
            st.session_state.selected_dataset_path = manual_path
            model_manager.update_dataset_path(manual_path)
            st.success("Path updated!")
            st.rerun()
    
    # Dataset validation (common for both modes)
    st.markdown("---")
    st.subheader("✅ Dataset Status")
    
    # Validate selected dataset
    is_valid, message = model_manager.validate_dataset(st.session_state.selected_dataset_path)
    
    if is_valid:
        st.success("✅ Valid Dataset")
        st.info(message)
        
        # Show current dataset path
        st.code(f"Path: {st.session_state.selected_dataset_path}")
        
        # Show dataset details
        try:
            import yaml
            with open(st.session_state.selected_dataset_path, 'r') as f:
                dataset_info = yaml.safe_load(f)
            
            with st.expander("📋 Dataset Details"):
                st.write(f"**Classes:** {dataset_info.get('nc', 'Unknown')}")
                st.write(f"**Names:** {dataset_info.get('names', [])}")
                st.write(f"**Train:** {dataset_info.get('train', 'Not specified')}")
                st.write(f"**Val:** {dataset_info.get('val', 'Not specified')}")
        except Exception:
            pass
    else:
        st.error("❌ Invalid Dataset")
        st.error(message)
        st.code(f"Path: {st.session_state.selected_dataset_path}")
    
    st.markdown("---")
    
    # Training configuration
    st.subheader("⚙️ Training Config")
    train_epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=50)
    train_batch = st.number_input("Batch Size", min_value=1, max_value=64, value=8)
    train_imgsz = st.selectbox("Image Size", [416, 480, 640, 800], index=2)
    train_device = st.selectbox("Device", ["0", "cpu"], index=0)

# Load models data
if st.button("🔄 Refresh Models"):
    st.session_state.models_data = model_manager.scan_models()

if not st.session_state.models_data:
    st.session_state.models_data = model_manager.scan_models()

# Models overview
if not st.session_state.models_data:
    st.warning("⚠️ No models found!")
    st.info("Please ensure you have a base model at `./Model/yolo11x.pt`")
else:
    st.header("📚 Available Models")
    
    # Create model cards
    for idx, model in enumerate(st.session_state.models_data):
        with st.expander(f"🎯 {model['type']} - {model['modified_str']}", 
                        expanded=(st.session_state.selected_model == model['id'])):
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**Path:** `{model['path']}`")
                st.write(f"**Size:** {model['size_mb']} MB")
            
            with col2:
                if model['metrics']:
                    st.write("**📊 Performance:**")
                    metrics = model['metrics']
                    if 'map50' in metrics:
                        st.write(f"mAP@50: {metrics['map50']:.3f}")
                    if 'final_loss' in metrics:
                        st.write(f"Loss: {metrics['final_loss']:.4f}")
                    if 'epochs' in metrics:
                        st.write(f"Epochs: {metrics['epochs']}")
            
            with col3:
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
    dataset_valid, dataset_message = model_manager.validate_dataset(st.session_state.selected_dataset_path)
    
    if selected_model_data and dataset_valid:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"**Selected Model:** {selected_model_data['type']}")
            st.code(f"Model: {selected_model_data['path']}")
            st.code(f"Dataset: {st.session_state.selected_dataset_path}")
            
            config_summary = f"""
📋 Training Configuration:
• Epochs: {train_epochs}
• Batch Size: {train_batch}  
• Image Size: {train_imgsz}
• Device: {train_device}
"""
            st.text(config_summary)
        
        with col2:
            if st.button("🚀 Start Training", type="primary", key="start_training"):
                training_config = {
                    'data': st.session_state.selected_dataset_path,
                    'epochs': train_epochs,
                    'batch': train_batch,
                    'imgsz': train_imgsz,
                    'device': train_device,
                    'workers': 0,
                    'patience': 30
                }
                
                st.info("🎓 Training started! This will take some time...")
                
                with st.spinner("🔄 Training in progress... Please wait."):
                    results, error = train_model(selected_model_data['path'], training_config)
                    
                    if error:
                        st.error(f"❌ Training failed: {error}")
                    else:
                        st.success("✅ Training completed successfully!")
                        
                        if hasattr(results, 'save_dir'):
                            st.success(f"📁 Results saved to: {results.save_dir}")
                            best_model = os.path.join(results.save_dir, "weights", "best.pt")
                            if os.path.exists(best_model):
                                st.success(f"🏆 New best model: {best_model}")
                        
                        # Refresh models
                        st.session_state.models_data = model_manager.scan_models()
                        st.info("🔄 Model list updated!")
                        st.rerun()
    
    elif selected_model_data and not dataset_valid:
        st.error("❌ Cannot start training - Dataset validation failed")
        st.error(dataset_message)

# Footer
st.markdown("---")
st.markdown("*💡 Tip: Training will run like the normal train.py script. The interface will show completion status.*")