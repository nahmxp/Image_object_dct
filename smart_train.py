from ultralytics import YOLO
import os
import glob
from datetime import datetime

def find_latest_model():
    """
    Find the latest trained model to resume from.
    Priority: latest fine-tuned model > base model
    """
    
    # Paths to check for models
    base_model_path = "./Model/yolo11x.pt"
    runs_dir = "./runs/detect"
    
    latest_model = None
    latest_time = 0
    
    print("🔍 Searching for existing models...")
    
    # Check if runs directory exists
    if os.path.exists(runs_dir):
        # Find all training runs
        train_dirs = glob.glob(os.path.join(runs_dir, "train*"))
        
        for train_dir in train_dirs:
            weights_dir = os.path.join(train_dir, "weights")
            
            # Check for best.pt and last.pt in each training run
            for model_file in ["best.pt", "last.pt"]:
                model_path = os.path.join(weights_dir, model_file)
                
                if os.path.exists(model_path):
                    # Get modification time
                    mod_time = os.path.getmtime(model_path)
                    
                    if mod_time > latest_time:
                        latest_time = mod_time
                        latest_model = model_path
                        
                    print(f"📁 Found: {model_path}")
    
    # If no fine-tuned model found, use base model
    if latest_model is None:
        if os.path.exists(base_model_path):
            latest_model = base_model_path
            print(f"📦 Using base model: {base_model_path}")
        else:
            raise FileNotFoundError(f"❌ No model found! Please ensure {base_model_path} exists.")
    else:
        # Convert to relative path for display
        rel_path = os.path.relpath(latest_model)
        mod_time_str = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M:%S")
        print(f"🎯 Using latest fine-tuned model: {rel_path}")
        print(f"📅 Last modified: {mod_time_str}")
    
    return latest_model

def train_model():
    """
    Train the model using the latest available checkpoint
    """
    
    # Training configuration
    config = {
        "data": "./Dataset/yolov11/dataset.yaml",  # Updated path based on your structure
        "imgsz": 640,
        "batch": 8,
        "epochs": 100,
        "workers": 0,
        "device": 0,
        "patience": 50,  # Early stopping patience
        "save": True,
        "save_period": 10,  # Save every 10 epochs
        "cache": False,  # Set to True if you have enough RAM
        "amp": True,  # Automatic Mixed Precision for faster training
    }
    
    try:
        # Find the best model to start from
        model_path = find_latest_model()
        
        print(f"\n🚀 Initializing YOLO model from: {model_path}")
        model = YOLO(model_path)
        
        print(f"\n📊 Training Configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎓 Starting training...")
        print("=" * 50)
        
        # Start training
        results = model.train(**config)
        
        print("=" * 50)
        print("✅ Training completed successfully!")
        
        # Display results information
        if hasattr(results, 'save_dir'):
            print(f"📁 Results saved to: {results.save_dir}")
            print(f"🏆 Best model: {os.path.join(results.save_dir, 'weights', 'best.pt')}")
            print(f"📄 Last model: {os.path.join(results.save_dir, 'weights', 'last.pt')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise

def main():
    """
    Main training function with error handling
    """
    
    print("🎯 YOLO Model Training Script")
    print("=" * 40)
    
    # Check if dataset exists
    dataset_path = "./Dataset/yolov11/dataset.yaml"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure your dataset.yaml file exists at the correct location.")
        return
    
    print(f"✅ Dataset found: {dataset_path}")
    
    try:
        # Run training
        results = train_model()
        print("\n🎉 Training pipeline completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()