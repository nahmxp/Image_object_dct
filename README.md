# YOLO Object Detection Project

This project implements object detection using YOLOv11 to detect and classify specific products: "7 OILS IN ONE 100 ML" and "BORO PLUS ANTIBACTERIAL SOAP 100 GM".

## Project Structure

```
Image_object_dct/
├── train.py              # Main training script
├── .gitignore            # Git ignore file
├── README.md             # This file
├── Dataset/             # Dataset folder
│   └── yolov11/         # Dataset configuration
│       ├── dataset.yaml # YOLO dataset configuration
│       ├── classes.txt  # Class names
│       ├── train/       # Training images and labels (not in git)
│       ├── val/         # Validation images and labels (not in git)
│       └── test/        # Test images and labels (not in git)
├── Model/               # Pre-trained models (not in git)
└── runs/                # Training results (not in git)
```

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