from ultralytics import YOLO, settings
import ultralytics
from pathlib import Path


def model_training():
    ultralytics.settings.update({'datasets_dir': "./dataset"})
    save_path = "/tmp/runs/detect/train2"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    model.train(data="datasets/data/dataset.yaml", epochs=50, imgsz=384, batch = 2, device="cpu", save_dir =save_path)





