import cv2
from PIL import Image
import numpy as np
import os
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalyst():
    def __init__(self,source_route, output_route, annotations_route):
        #ROUTE THAT I AM GOING TO USE TO READ/WRITE ON FOLDERS
        self.route = source_route
        self.output_folder = output_route
        self.annotations_route = annotations_route

        #LIST THAT I USE TO STORE TRAINING LABELS
        self.training_features = []
        self.training_labels = []
        self.elements_names = []

        #LIST THAT I USE TO IDENTIFY CLASSES THAT WILL NEED MORE DATA AUGMENTATION
        self.dc_circuits = []
        self.ac_circuits = []

        #DICTIONARY TO ENCODE LABELS
        self.names_dict = {
            'Resistor' : 0,
            'Capacitor': 1,
            'Inductor' : 2,
            'DC voltage source': 3,
            'AC voltage source': 4
        }

    def image_size_searching(self):
        """THIS FUNCTION IS GOING TO IDENTIFY THE MEAN WIDTH AND HEIGHT OF THE IMAGES
        THAT WAY I CAN DEFINE A CORRECT SIZE TO RESIZE THESE IMAGES"""
        self.original_width = []
        self.original_height = []

    
        for file in os.listdir(self.route):
            if file.lower().endswith(".jpg"):
                image_path = os.path.join(self.route,file)
                try:
                    image = cv2.imread(image_path)
                except Exception as e:
                    print(f"Error while trying to open an image: {e}")
            
            self.original_width.append(image.shape[1])
            self.original_height.append(image.shape[0])
    
        self.width_mean = np.mean(self.original_width)
        self.height_mean = np.mean(self.original_height)

        self.image_resize_save()

    def image_resize_save(self):
        """THIS FUNCTION IS USED TO RESIZE THE DATASET TO THE IMAGE SIZES DEFINED BEFORE"""
        #new_size = (int(self.width_mean), int(self.height_mean))
        self.new_size = (224,224)

        for file in os.listdir(self.route):
            file_route = os.path.join(self.route,file)
            try:
                image = cv2.resize(cv2.imread(file_route),self.new_size, interpolation=cv2.INTER_AREA)
                output_path = os.path.join(self.output_folder, file)
                cv2.imwrite(output_path, image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                self.training_features.append(image)

            except Exception as e:
                print(f"Error while trying to resize and save and image {e}")

    def xml_data_extractor(self):
        """THIS IS THE FUNCTION THAT WILL EXTRACT LABELS DATA FROM THE XML FILES
           THESE XML FILES DESCRIBE THE BOUNDING BOXES AROUND THE CIRCUIT ELEMENTS"""
        for idx,file in enumerate(os.listdir(self.annotations_route)):
            file_labels = []
            file_path = os.path.join(self.annotations_route, file)
            try:
                with open(file_path, 'r') as f:
                    file_data = f.read()
                    beauti_data = BeautifulSoup(file_data, 'lxml-xml')
                    elements_list = beauti_data.find_all('object')
                
                    for element in elements_list:
                        name = element.find("name").text
                        if name =="Capactitor":
                            name ="Capacitor"
                        elif name == "DC voltage source":
                            self.dc_circuits.append(idx)
                        elif name == "AC voltage source":
                            self.ac_circuits.append(idx)

                        self.elements_names.append(name)
                        coord_x1 = element.find("xmin").text
                        coord_x2 = element.find("xmax").text
                        coord_y1 = element.find("ymin").text
                        coord_y2 = element.find("ymax").text
                        file_labels.append([float(coord_x1), float(coord_x2), float(coord_y1), float(coord_y2), float(self.names_dict[name])])
                self.training_labels.append(file_labels)
            except Exception as e:
                print(f"Error while reading xml files and getting their information {e}")


    def dataset_info_summary(self):
        """HERE I AM GOING TO VISUALIZE SOME DATASET INFORMATION, TYPICAL HEIGHT AN WIDTH, CLASS DISTRIBUTION"""
        elements_dataframe = pd.DataFrame(self.elements_names)
        elements_names =elements_dataframe.value_counts()
        print(elements_names)
        fig,axes = plt.subplots(1,1)
        axes.pie(elements_names, labels= elements_names.index, autopct='%1.2f%%', colors=['gold', 'skyblue', 'lightgreen', 'red', 'green'], startangle=90)
        plt.show()

    def xml_data_builder(self,mode,anotations_route):
        """THIS IS THE FUNCTION THAT WILL EXTRACT LABELS DATA FROM THE XML FILES
           THESE XML FILES DESCRIBE THE BOUNDING BOXES AROUND THE CIRCUIT ELEMENTS"""

        route = "./datasets/data/dataset/images/train"
        label_route = "./datasets/data/dataset/labels/train"
        route = route.replace("train", mode)
        label_route = label_route.replace("train", mode)

        for file in os.listdir(route):
            if file.endswith(".jpg"):
                xml_file = os.path.join(anotations_route, file)
                xml_file = os.path.splitext(xml_file)[0] + ".xml"
                xml_results = self.xml_data_getter(xml_file)
                final_path = os.path.join(label_route, os.path.splitext(file)[0])

                with open(final_path, 'w') as f:

                    for item in xml_results:
                        x_center, y_center, width, height = self.get_coordinates([int(item[1]), int(item[2]), int(item[3]), int(item[4])], (224,2))
                        new_line = f"{item[0]} {x_center} {y_center} {width} {height}\n"
                        f.write(new_line)
                    f.close()




    def xml_data_getter(self, file_path):
        """THIS IS THE FUNCTION THAT WILL EXTRACT LABELS DATA FROM THE XML FILES
           THESE XML FILES DESCRIBE THE BOUNDING BOXES AROUND THE CIRCUIT ELEMENTS"""
        try:
            with open(file_path, 'r') as f:
                result_list = []
                file_data = f.read()
                beauti_data = BeautifulSoup(file_data, 'lxml-xml')
                elements_list = beauti_data.find_all('object')

                for element in elements_list:
                    name = element.find("name").text
                    if name == "Capactitor":
                        name = "Capacitor"

                    self.elements_names.append(name)
                    coord_x1 = element.find("xmin").text
                    coord_x2 = element.find("xmax").text
                    coord_y1 = element.find("ymin").text
                    coord_y2 = element.find("ymax").text
                    name = self.names_dict[name]
                    coord_lines = [name,coord_x1, coord_x2, coord_y1, coord_y2]
                    result_list.append(coord_lines)
                return result_list
        except Exception as e:
            print(f"Error while reading xml files and getting their information {e}")

    def get_coordinates(self,bbox_tensor, image_size):
        """THIS FUNCTION WILL PREPARE THE COORDINATES THEY WAY YOLO EXPECTS IT
        FIRST GET COORDINATES NORMALIZED, SINCE YOLO EXPECTS TO BE BETWEEN 0 AND 1
        THIS IS BECAUSE YOLO RESEARCHES REALISED THAT IT WAYS EASIER TO USE OFFSETS OF GRID CELLS RATHER THAN PURE COORDINATES"""
        normalized_width = 1.0 / image_size[0]
        normalized_height = 1.0 / image_size[1]

        width = (bbox_tensor[1] - bbox_tensor[0]) * normalized_width
        height = (bbox_tensor[3] - bbox_tensor[2]) * normalized_height

        # GET X AND Y AXIS CENTERS
        x_center = ((bbox_tensor[1] + bbox_tensor[0]) / 2.0) * normalized_width
        y_center = ((bbox_tensor[3] + bbox_tensor[2]) / 2.0) * normalized_height

        return x_center, y_center, width, height



