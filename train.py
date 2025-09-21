from ultralytics import YOLO
model = YOLO("./Model/yolo11m-seg.pt")

model.train(data = "./Dataset/ROBO/data.yaml", imgsz = 640, batch = 8, epochs = 300, workers = 0, device = 0)  # load a pretrained model (recommended for training)