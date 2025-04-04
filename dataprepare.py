"""THIS FILE IS PART OF THE FIRST APPROXIMATION THAT I DID TO THE PROBLEM USING DATASET CLASS AND ITERATING OVER IT USING A DATALOADER
   BUT  IT IS  NOT USED IN THE FINAL SOLUTION AFTER REALISING THAT IT WOULDN'T WORK IN ORDER TO FINE TUNE YOLOV8 FROM ULTRALYTICS
   I KEEP IT HERE SO YOU CAN SEE HOW I USE THIS CLASSES"""

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from data_analysis import DataAnalyst
import torch
import os
import shutil

class CircuitDataset(Dataset):
    def __init__(self, training_images , training_labels,training_route,image_size):

        #INITIALIZE TRAINING FEATURES AND LABELS
        self.features = training_images
        self.labels = training_labels

        #INITIALIZE TRANSFORMATIONS
        self.transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear = 10),
            transforms.RandomPerspective(distortion_scale=0.5, p = 0.5),
            transforms.ToTensor()

        ])
        self.training_route = training_route
        self.image_size = image_size
        self.yolo_train_labels = "./dataset/data/Training/Label"

    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        print(f"Imagen nº {idx}")
        """THIS FUNCTION IS GOING TO ITERATE OVER EACH EXAMPLE IN MY DATASET APPLYING DIFFERENT TRASNFORMATIONS
        PREPARING EACH TRAINING EXAMPLE THE WAY THAT THE MODELS EXPECTS IT IN ORDER TO FINE TUNE"""

        #FEATURE PROCESSING
        features_iter = self.features[idx]
        feature_transformed = self.transforms(features_iter)

        #LABEL PROCESSING
        labels_iter = self.labels[idx]
        bbox_tensor = torch.tensor([bbox[:4] for bbox in labels_iter], dtype=torch.float32)
        label_tensor = torch.tensor([bbox[4] for bbox in labels_iter], dtype=torch.int64)
        label_dict = {
            "bboxes": bbox_tensor,
            "labels": label_tensor
        }
        
        file = os.listdir(self.training_route)[idx]
        new_file_route = os.path.join(self.yolo_train_labels, file)
        new_file_route = os.path.splitext(new_file_route)[0]+ ".txt"
        with open(new_file_route, 'w', ) as f:
            for index,bbox in enumerate(label_dict['bboxes']):
                x_center, y_center, width, height = get_coordinates(bbox ,self.image_size)
                new_line = f"{label_dict['labels'][index]} {x_center} {y_center} {width} {height}\n"
                f.write(new_line)
        return feature_transformed, label_dict

def data_asociation(mode):
    """THIS FUNCTION IS GOING TO PREPARE THE dataset FOLDER
    THIS FOLDER HAS THE STRUCTURE THAT MODELS LIKE YOLOV8 EXPECTS IN ORDER TO FINE TUNE"""
    images = "./dataset/data/dataset/images/train"
    labels = "./dataset/data/dataset/labels/train"

    images = images.replace("train", mode)
    labels = labels.replace("train", mode)

    all_labels = "./dataset/data/Training/Label"
    for file in os.listdir(images):
        if file.endswith(".jpg"):
            new_file = os.path.join(all_labels, file)
            new_file = os.path.splitext(new_file)[0] + ".txt"
            shutil.copy(new_file, labels)


def analyst_starting2(route,output_folder,annotations_route):
    data_analyst = DataAnalyst(route, output_folder, annotations_route)

    data_analyst.xml_data_builder("train", annotations_route)
    data_analyst.xml_data_builder("val",annotations_route)
    data_analyst.xml_data_builder("test",annotations_route)


route = "./dataset/data/Training/Images"
output_folder = "./data/Images"
annotations_route = "./dataset/data/Annotations/"
analyst_starting2(route,output_folder,annotations_route)