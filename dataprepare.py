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
            transforms.ToTensor(),
        ])
        self.training_route = training_route
        self.image_size = image_size
        self.yolo_train_labels = "./data/Training/Label"
        self.yolo_val_labels = "./data/dataset/labels/val"

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
    images = "./data/dataset/Images/train"
    labels = "./data/dataset/labels/train"

    images = images.replace("train", mode)
    labels = labels.replace("train", mode)

    print(images)
    print(labels)
    all_labels = "./data/Training/Label"
    for file in os.listdir(images):
        if file.endswith(".jpg"):
            new_file = os.path.join(all_labels, file)
            new_file = os.path.splitext(new_file)[0] + ".txt"
            shutil.copy(new_file, labels)



def get_coordinates(bbox_tensor,image_size):
    """THIS FUNCTION WILL PREPARE THE COORDINATES THEY WAY YOLO EXPECTS IT
    FIRST GET COORDINATES NORMALIZED, SINCE YOLO EXPECTS TO BE BETWEEN 0 AND 1
    THIS IS BECAUSE YOLO RESEARCHES REALISED THAT IT WAYS EASIER TO USE OFFSETS OF GRID CELLS RATHER THAN PURE COORDINATES"""
    normalized_width = 1.0/image_size[0]
    normalized_height = 1.0/image_size[1]


    width = (bbox_tensor[1] - bbox_tensor[0]) * normalized_width
    height = (bbox_tensor[3] - bbox_tensor[2]) * normalized_height

    #GET X AND Y AXIS CENTERS
    x_center = ((bbox_tensor[1] + bbox_tensor[0])/2.0) * normalized_width
    y_center = ((bbox_tensor[3] + bbox_tensor[2])/2.0) * normalized_height

    return x_center, y_center, width, height


def analyst_starting(route, output_folder, annotations_route):
    #THIS FUNCTION IS GOING TO BE CALLED BY THE MAIN IN ORDER TO START THE JOB
    data_analyst = DataAnalyst(route, output_folder, annotations_route)
    data_analyst.image_size_searching()
    data_analyst.xml_data_extractor()
    data_analyst.dataset_info_summary()
    #dataset = CircuitDataset(data_analyst.training_features, data_analyst.training_labels, data_analyst.route,data_analyst.new_size)
    #training_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    #for x,y in training_loader:
    #    print("Imprimo imagen")

analyst_starting()