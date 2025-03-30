import dataprepare
from ultralytics import YOLO
#data_analyst.dataset_info_summary()

route = "./data/circuits/"
output_folder = "./data/Images"
annotations_route = "./data/Annotations"
dataprepare.analyst_starting(route, output_folder, annotations_route)
dataprepare.data_asociation("train")
dataprepare.data_asociation("val")
dataprepare.data_asociation("test")

