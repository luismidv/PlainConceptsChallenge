import os
def xml_data_extractor(mode, anotations_route):
    """THIS IS THE FUNCTION THAT WILL EXTRACT LABELS DATA FROM THE XML FILES
       THESE XML FILES DESCRIBE THE BOUNDING BOXES AROUND THE CIRCUIT ELEMENTS"""

    route = "./datasets/data/dataset/images/train"
    label_route = "./datasets/data/dataset/labels/train"
    route = route.replace("train", mode)
    label_route = label_route.replace("train", mode)

    for file in os.listdir(route):
        xml_file = os.path.join(anotations_route, file)
        xml_file = os.path.splitext(xml_file)[0] + ".xml"
        xml_results = xml_data_extractor(xml_file)
        final_path = os.path.join(label_route, file)

anot_route = "./dataset/data/Annotations"
xml_data_extractor(mode = "train",anotations_route=anot_route)
xml_data_extractor("test",anot_route)
xml_data_extractor("val",anot_route)

