from ultralytics import YOLO
model = YOLO("./Model/yolo11x.pt")

model.train(data = "./Dataset/yolov11/dataset.yaml", imgsz = 640, batch = 8, epochs = 100, workers = 0, device = 0)  # load a pretrained model (recommended for training)