# 🎯 YOLO Object Detection System

A comprehensive YOLO-based object detection system with multiple training interfaces, real-time detection capabilities, and dataset management tools.

## � Project Structure

```
E:\NDEV\Image_object_dct\
├── 📄 Python Scripts (.py files)
├── 📂 Dataset/                    # Training datasets (ignored in git)
├── 📂 Model/                      # Pre-trained models (ignored in git)
├── 📂 runs/                       # Training outputs (ignored in git)
├── 📂 tools/                      # Utility scripts
└── 📋 README.md                   # This file
```

## 🐍 Python Files Overview

### 🎯 **Core Training Scripts**

#### 1. **`train.py`** - Basic Training Script
**Purpose:** Simple YOLO model training with minimal configuration
```python
# Basic YOLO training
from ultralytics import YOLO
model = YOLO("./Model/yolo11x-seg.pt")
model.train(data="./Dataset/YOLO/yolov11/dataset.yaml", epochs=100)
```
**Usage:**
```bash
python train.py
```

#### 2. **`smart_train.py`** - Intelligent Auto-Resume Training
**Purpose:** Advanced training with automatic resume, model comparison, and performance tracking
- ✅ Auto-detects and resumes interrupted training
- ✅ Compares multiple model performances
- ✅ Smart model selection and validation
- ✅ Comprehensive logging and monitoring

**Usage:**
```bash
python smart_train.py
```

### 🎛️ **Training Management Interfaces**

#### 3. **`training_manager.py`** - Basic Training GUI
**Purpose:** Simple Streamlit interface for model training management
- 📊 Model browser and selection
- ⚙️ Basic training configuration
- 🎯 Simple training execution

**Usage:**
```bash
streamlit run training_manager.py
```

#### 4. **`training_manager_live.py`** - Advanced Training Manager
**Purpose:** Advanced Streamlit interface with real-time monitoring
- 📊 Live training progress tracking
- 📈 Real-time metrics visualization
- 🔄 Advanced model comparison
- 📋 Comprehensive training logs

**Usage:**
```bash
streamlit run training_manager_live.py
```

#### 5. **`training_manager_simple.py`** - Enhanced Simple Manager
**Purpose:** Streamlined training interface with dataset browsing
- 📁 File browser for dataset selection
- ✅ Dataset validation and verification
- 🎯 Simplified training workflow
- 🔄 Auto-refresh model list

**Usage:**
```bash
streamlit run training_manager_simple.py
```

### 🔍 **Detection Applications**

#### 6. **`test_detection.py`** - Image Testing Interface
**Purpose:** Simple image upload and detection testing
- � Image upload functionality
- 🎯 Quick detection testing
- 📊 Results visualization
- 💾 Save detection results

**Usage:**
```bash
streamlit run test_detection.py
```

#### 7. **`app.py`** - Full Detection Application
**Purpose:** Comprehensive detection system with multiple input sources
- 📷 Real-time camera detection
- 📤 Image upload and batch processing
- 🎥 Video file processing
- � Advanced visualization options
- 💾 Export and save capabilities

**Usage:**
```bash
streamlit run app.py
```

#### 8. **`camera.py`** - Real-time Camera Detection
**Purpose:** Dedicated webcam real-time object detection
- 📷 Live webcam feed processing
- 🎯 Real-time YOLO inference
- 📊 Live bounding box visualization
- ⚙️ Configurable model selection

**Usage:**
```bash
streamlit run camera.py
```
### 🛠️ **Utility Scripts**

#### 9. **`tools/augment_yolov11.py`** - Dataset Augmentation Tool
**Purpose:** Advanced dataset augmentation for YOLO training
- 🔄 Multiple augmentation techniques (flip, rotate, shear, noise)
- 📊 YOLO label preservation and transformation
- 📁 Batch processing capabilities
- ⚙️ Configurable augmentation parameters

**Features:**
- Horizontal/Vertical flips
- 90° rotations (clockwise, counter-clockwise, upside down)
- Small angle rotations (±7°)
- Shear transformations (±10° H/V)
- Grayscale conversion (15% of images)
- Noise addition (Gaussian + Salt&Pepper)
- Automatic resize to 640×640

**Usage:**
```bash
python tools/augment_yolov11.py --dataset-yaml "./Dataset/YOLO/yolov11/dataset.yaml" --out-root "./Dataset/YOLO/yolov11_aug" --include-original
```

## 📦 Dependencies & Installation

### **System Requirements**
- **Python:** 3.8+ (Tested on 3.12)
- **CUDA:** Compatible GPU (RTX 4060 Ti recommended)
- **RAM:** 8GB+ (16GB recommended)
- **Storage:** 10GB+ free space

### **Required Dependencies**

#### **Core ML Libraries**
```bash
pip install ultralytics torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### **Computer Vision & Image Processing**
```bash
pip install opencv-python opencv-python-headless pillow
```

#### **Data Science & Visualization**
```bash
pip install numpy pandas matplotlib seaborn plotly
```

#### **Web Interface (Streamlit)**
```bash
pip install streamlit streamlit-webrtc
```

#### **Data Augmentation**
```bash
pip install albumentations
```

#### **Utilities**
```bash
pip install pyyaml tqdm psutil
```

### **One-Command Installation**
```bash
pip install ultralytics torch torchvision torchaudio opencv-python pillow numpy pandas matplotlib seaborn plotly streamlit streamlit-webrtc albumentations pyyaml tqdm psutil --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start Guide

### **1. Environment Setup**
```bash
# Clone repository
git clone <your-repo-url>
cd Image_object_dct

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### **2. Dataset Preparation**
```bash
# Your dataset should follow this structure:
Dataset/
├── YOLO/yolov11/
│   ├── data.yaml          # Dataset configuration
│   ├── train/
│   │   ├── images/        # Training images
│   │   └── labels/        # YOLO format labels
│   ├── val/
│   │   ├── images/        # Validation images
│   │   └── labels/        # YOLO format labels
│   └── test/              # Optional test set
```

### **3. Model Setup**
```bash
# Download YOLO models (will be downloaded automatically on first run)
# Models are saved in: ./Model/
```

### **4. Usage Examples**

#### **Basic Training**
```bash
python train.py
```

#### **Smart Training with Auto-Resume**
```bash
python smart_train.py
```

#### **GUI Training Manager**
```bash
streamlit run training_manager_simple.py
```

#### **Real-time Detection**
```bash
streamlit run camera.py
```

#### **Full Detection App**
```bash
streamlit run app.py
```

#### **Dataset Augmentation**
```bash
python tools/augment_yolov11.py --dataset-yaml "./Dataset/YOLO/yolov11/dataset.yaml" --out-root "./Dataset/YOLO/yolov11_aug"
```

## 🎯 Workflow Recommendations

### **For Beginners:**
1. **Start with:** `training_manager_simple.py` - Easy GUI interface
2. **Test models:** `test_detection.py` - Simple image testing
3. **Real-time:** `camera.py` - Webcam detection

### **For Advanced Users:**
1. **Training:** `smart_train.py` - Auto-resume and comparison
2. **Monitoring:** `training_manager_live.py` - Real-time progress
3. **Production:** `app.py` - Full-featured detection system

### **For Dataset Management:**
1. **Augmentation:** `tools/augment_yolov11.py` - Expand dataset
2. **Training:** Any training script with augmented data
3. **Validation:** Compare models with original vs augmented datasets

## ⚙️ Configuration

### **Training Parameters (Common)**
- **Image Size:** 640x640 (adjustable)
- **Batch Size:** 8 (reduce if memory issues)
- **Epochs:** 100 (increase for better accuracy)
- **Workers:** 0 (Windows) / 4 (Linux) for data loading

### **Model Options**
- **YOLOv11n:** Fastest, lowest accuracy
- **YOLOv11s:** Balanced speed/accuracy
- **YOLOv11m:** Medium size, good accuracy
- **YOLOv11l:** Large, high accuracy
- **YOLOv11x:** Largest, highest accuracy

## 🔧 Troubleshooting

### **Common Issues**
1. **CUDA not available:** Install CUDA-compatible PyTorch
2. **Memory errors:** Reduce batch size or image size
3. **Multiprocessing errors (Windows):** Set `workers=0`
4. **Streamlit port busy:** Use `--server.port 8502`

### **Performance Tips**
- Use GPU for training (`device=0`)
- Monitor GPU memory usage
- Use appropriate batch size for your GPU
- Consider mixed precision training (enabled by default)

## 📊 Output Structure

```
runs/
├── detect/               # Object detection results
│   ├── train*/          # Training sessions
│   └── predict*/        # Prediction results
└── segment/              # Segmentation results (if using -seg models)
    ├── train*/          # Training sessions
    └── predict*/        # Prediction results
```

## 🎯 Dataset Classes

### **Current Project Supports:**

#### **Main Products (7 classes):**
- Boro Plus Antiseptic Moisturising Soap
- Emami 7 oils in one
- Garnier Men turbo face wash
- He Perfume
- Himalaya Men face wash
- navratna oil bottle
- navratna oil box

#### **Additional Product Classes (4 classes):**
- Fogg Absolute
- Fogg Dynamic
- Fogg Extreme
- Fogg Punch

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLO** - Core detection framework
- **Streamlit** - Web interface framework
- **OpenCV** - Computer vision operations
- **Albumentations** - Data augmentation library

---

**🎯 Ready to detect objects? Start with `streamlit run training_manager_simple.py` for the easiest experience!**



## 🎯 Interface Comparison

| Feature | train.py | smart_train.py | test_detection.py | app.py | training_manager_live.py |
|---------|----------|----------------|-------------------|--------|--------------------------|
| **Training** | ✅ Basic | ✅ Smart | ❌ No | ❌ No | ✅ Advanced |
| **Auto-resume** | ❌ No | ✅ Yes | ❌ No | ❌ No | ✅ Yes |
| **GUI Interface** | ❌ No | ❌ No | ✅ Simple | ✅ Full | ✅ Advanced |
| **Real-time Progress** | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Model Comparison** | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |
| **Image Testing** | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Camera Support** | ❌ No | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Background Training** | ❌ No | ❌ No | ❌ No | ❌ No | ✅ Yes |

## 🛠️ Development Workflow

### 🔄 Typical Development Cycle

1. **Initial Training**:
   ```powershell
   python smart_train.py  # First training run
   ```

2. **Model Testing**:
   ```powershell
   streamlit run test_detection.py  # Test with images
   ```

3. **Continue Training**:
   ```powershell
   streamlit run training_manager_live.py  # Advanced training
   ```

4. **Production Testing**:
   ```powershell
   streamlit run app.py  # Full detection system
   ```

### 🐛 Debugging & Monitoring

- **Training Logs**: Check `runs/detect/train*/` for detailed logs
- **Model Metrics**: View in `results.csv` and `results.png`
- **Real-time Monitoring**: Use `training_manager_live.py`
- **Error Handling**: All scripts include comprehensive error handling

## 📈 Performance Monitoring

### 📊 Available Metrics

- **mAP@50**: Mean Average Precision at IoU 0.5
- **mAP@50-95**: Mean Average Precision at IoU 0.5-0.95
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Box Loss**: Bounding box regression loss
- **Class Loss**: Classification loss

### 🎯 Model Selection Criteria

**For Training Continuation:**
- Use **`best.pt`** for optimal performance
- Use **`last.pt`** for latest state (if training interrupted)

**For Inference:**
- Always use **`best.pt`** for production
- Lower confidence threshold for more detections
- Higher confidence threshold for fewer false positives

## 🚀 Advanced Features

### 🔥 Smart Training System
- **Automatic Model Detection**: Scans all training runs
- **Timestamp-based Selection**: Uses newest model automatically
- **Fallback Mechanism**: Base model if no fine-tuned version
- **Progress Tracking**: Real-time epoch and loss monitoring

### 🎮 Live Training Manager
- **Multi-model Comparison**: Side-by-side performance metrics
- **Interactive Selection**: Click to choose training base model
- **Real-time Streaming**: Live training logs and progress
- **Background Processing**: Non-blocking training execution

### 📱 Detection Interfaces
- **Multi-modal Support**: Images, videos, live camera
- **Confidence Control**: Real-time threshold adjustment
- **Performance Optimization**: Frame skipping, buffer management
- **Cross-platform**: Windows, Linux, macOS compatibility

## 🔍 Troubleshooting

### 🐛 Common Issues

**1. Camera Not Working:**
```powershell
# Issue: DirectShow/MSMF errors
# Solution: App includes multiple backend fallbacks
```

**2. Model Not Found:**
```powershell
# Issue: FileNotFoundError for model
# Solution: Ensure base model exists in Model/yolo11x.pt
```

**3. Dataset Path Error:**
```powershell
# Issue: Dataset not found
# Solution: Check dataset.yaml path configuration
```

**4. Training Interrupted:**
```powershell
# Issue: Training stops unexpectedly
# Solution: Use smart_train.py to auto-resume from best checkpoint
```

### 🛠️ System Requirements

- **Python**: 3.8+
- **GPU**: CUDA-compatible (recommended)
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB+ for models and datasets
- **Camera**: USB/built-in camera for real-time detection

## 🎯 Next Steps

### 🚀 Potential Enhancements

1. **Model Export**: ONNX/TensorRT conversion for deployment
2. **API Integration**: REST API for remote inference
3. **Data Augmentation**: Advanced augmentation pipeline
4. **Multi-class Expansion**: Support for more product categories
5. **Cloud Deployment**: Docker containerization and cloud hosting

### 📚 Learning Resources

- **YOLOv11 Documentation**: [Ultralytics Docs](https://docs.ultralytics.com/)
- **Streamlit Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
- **Computer Vision**: OpenCV and PIL libraries
- **Deep Learning**: PyTorch framework fundamentals

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv11 implementation
- **Streamlit** for the web interface framework
- **OpenCV** for computer vision capabilities
- **PyTorch** for the deep learning foundation

---

*💡 **Tip**: Start with `smart_train.py` for training and `test_detection.py` for testing. Use `training_manager_live.py` for advanced model management!*

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Image_object_dct
```

### 2. Create Python Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

Install the required Python packages:

```bash
# Install Ultralytics (includes YOLO)
pip install ultralytics

# Install PyTorch with CUDA support (for GPU training)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Note:** The CUDA 12.8 version is used above. If you have a different CUDA version, visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) to get the correct installation command.

### 5. Download Pre-trained Model

Create a `Model` folder and download the YOLOv11 model:

```bash
# Create Model directory
mkdir Model

# Download YOLOv11x model (you can also use yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt)
# Visit https://github.com/ultralytics/ultralytics and download yolo11x.pt
# Place the downloaded file in the Model/ directory
```

**Model Options:**
- `yolo11n.pt` - Nano (fastest, least accurate)
- `yolo11s.pt` - Small 
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (slowest, most accurate)

### 6. Prepare Dataset

#### Dataset Structure
Your dataset should follow this structure:

```
Dataset/
└── yolov11/
    ├── dataset.yaml         # Already provided
    ├── classes.txt          # Already provided
    ├── train/
    │   ├── images/          # Training images (.jpg, .png)
    │   └── labels/          # Training labels (.txt)
    ├── val/
    │   ├── images/          # Validation images
    │   └── labels/          # Validation labels
    └── test/
        ├── images/          # Test images
        └── labels/          # Test labels
```

#### Label Format
Each label file should contain bounding box annotations in YOLO format:
```
class_id center_x center_y width height
```

Where:
- `class_id`: 0 for "7 OILS IN ONE 100 ML", 1 for "BORO PLUS ANTIBACTERIAL SOAP 100 GM"
- `center_x, center_y`: Center coordinates (normalized 0-1)
- `width, height`: Box dimensions (normalized 0-1)

#### Adding Your Dataset
1. Place your training images in `Dataset/yolov11/train/images/`
2. Place corresponding label files in `Dataset/yolov11/train/labels/`
3. Place validation images in `Dataset/yolov11/val/images/`
4. Place corresponding label files in `Dataset/yolov11/val/labels/`
5. (Optional) Place test images in `Dataset/yolov11/test/images/`
6. (Optional) Place corresponding label files in `Dataset/yolov11/test/labels/`

### 7. Configuration Files

The project includes pre-configured files:

#### `Dataset/yolov11/dataset.yaml`
```yaml
path: ./Dataset/yolov11
train: train
val: val
test: test

nc: 2
names: ['7 OILS IN ONE 100 ML', 'BORO PLUS ANTIBACTERIAL SOAP 100 GM']
```

#### `Dataset/yolov11/classes.txt`
```
7 OILS IN ONE 100 ML
BORO PLUS ANTIBACTERIAL SOAP 100 GM
```

### 8. Start Training

Run the training script:

```bash
python train.py
```

#### Training Parameters
The current configuration in `train.py`:
- **Model**: YOLOv11x (extra large)
- **Image size**: 640x640
- **Batch size**: 8
- **Epochs**: 100
- **Workers**: 0 (for Windows compatibility)
- **Device**: 0 (GPU 0, change to 'cpu' for CPU training)

#### Modify Training Parameters
You can edit `train.py` to customize training:

```python
from ultralytics import YOLO

model = YOLO("./Model/yolo11x.pt")

model.train(
    data="./Dataset/yolov11/dataset.yaml",
    imgsz=640,          # Image size
    batch=8,            # Batch size (reduce if GPU memory issues)
    epochs=100,         # Number of training epochs
    workers=0,          # Number of data loading workers
    device=0            # GPU device (0, 1, 2...) or 'cpu'
)
```

### 9. Monitor Training

Training results will be saved in the `runs/detect/train/` directory:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.png` - Training metrics plot
- `confusion_matrix.png` - Confusion matrix
- Various curve plots (P, R, F1, etc.)

### 10. System Requirements

#### Minimum Requirements:
- Python 3.8+
- 8GB RAM
- 10GB free disk space

#### Recommended for GPU Training:
- NVIDIA GPU with 6GB+ VRAM
- CUDA 11.8 or 12.x
- 16GB+ RAM
- 50GB+ free disk space

#### CPU Training:
- Multi-core CPU
- 16GB+ RAM
- Expect significantly longer training times

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**
   - Reduce batch size in `train.py`
   - Use a smaller model (yolo11n.pt instead of yolo11x.pt)

2. **Dataset Not Found**
   - Ensure dataset.yaml path is correct
   - Check that images and labels are in the correct directories

3. **Permission Errors**
   - Run PowerShell as Administrator (Windows)
   - Check file permissions

4. **Import Errors**
   - Ensure virtual environment is activated
   - Reinstall packages: `pip install --upgrade ultralytics torch torchvision`

## Usage After Training

After training, you can use the trained model for inference:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Run inference
results = model('path/to/your/image.jpg')

# Display results
results[0].show()
```

## File Management

### What's Tracked in Git:
- Python scripts (`train.py`)
- Configuration files (`Dataset/yolov11/dataset.yaml`, `Dataset/yolov11/classes.txt`)
- Documentation (`README.md`)
- Git configuration (`.gitignore`)

### What's NOT Tracked in Git:
- Virtual environment (`venv/`)
- Model files (`*.pt`, `Model/`)
- Dataset images and labels
- Training results (`runs/`)
- Cache files

This keeps the repository lightweight while preserving all necessary code and configuration.