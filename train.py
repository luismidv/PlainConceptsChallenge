from ultralytics import YOLO, settings
import ultralytics
import torch
#DATA AUGMENTATION SECTION
from ultralytics.data.augment import Albumentations
from ultralytics.utils import LOGGER,colorstr
import albumentations as A
import cv2

def create_transformations(p=1.0):
  """THIS IS WHERE I CREATED THE FIRST TRASNFORMATIONS OVER THE IMAGE, FINALLY I AM JUST USING THE ULTRALYTICS DEFAULT ONE"""
  prefix = colorstr("albumentations: ")
  try:
    transforms = [
      A.RandomRain(p=0.1, slant_lower=-10, slant_upper=10,
                              drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                              blur_value=5, brightness_coefficient=0.9, rain_type="default"),
      A.Rotate(limit = 10, p=0.5),
      A.Blur(p=0.1),
      A.MedianBlur(p=0.1),
      A.ToGray(p=0.01),
      A.CLAHE(p=0.01),
      A.ImageCompression(quality_lower=75, p=0.0),
    ]
    transform = A.Compose(transforms, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
    return transform

  except Exception as e:
    print(f"Error while creating transformations {e}")

def model_training():
    """FUNCTION USED TO TRAIN THE MODEL"""
    try:
        transformations = create_transformations()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Available device: ", device)
        model = YOLO("yolov8n.pt")
        results = model.train(data="/datasets/dataset.yaml", epochs=50, imgsz=384, batch = 2, augment = True,device=device,verbose=True)

        return model
    except Exception as e:
        print(e)

def image_show_fn(bboxes,path):
    """THIS IS THE FUNCTION USING BY model_testing FUNCTION TO DRAW BBOXES"""
    image = cv2.imread(path)
    for bbox in bboxes:
        bbox = [int(num) for num in bbox]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.imshow(image)
    #THIS METHOD IS ONLY CALLABLE IN GOOGLE COLAB TYPES
    #cv2_imshow(image)

def model_testing(model, test_image):
    """THIS IS THE FUNCTION THAT THE MODEL WILL USE AT INFERENCE TIME
        IT LOADS THE BEST WEIGHTS, DO THE INFERENCE ON THE IMAGE, CALL THE
        FUNCTION THAT WILL DRAW THE BOUNDING BOXES AND FINALLY DRAW BBOXES """
    model = YOLO("/content/runs/detect/train12/weights/best.pt")
    image_results = model(test_image)
    for image in image_results:
        image_show_fn(image.boxes.xyxy,image.path)




