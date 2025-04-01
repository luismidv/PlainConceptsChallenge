from ultralytics import YOLO, settings
import ultralytics
from pathlib import Path
import os
import torch

def model_training():
    try:
        #ultralytics.settings.update({'datasets_dir': "./dataset"})
        #save_path = '/tmp/runs/detect/train2'
        #Path(save_path).mkdir(parents=True, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Available device: ", device)
        model = YOLO("yolov8n.pt")
        model.train(data="datasets/dataset.yaml", epochs=20, imgsz=244, batch = 2, device=device,verbose=True)

    except Exception as e:
        print(e)





