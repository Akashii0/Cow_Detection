# Cattle Detection, Akeem
from ultralytics import YOLO
import yaml
model = YOLO("yolov8n.yaml")
# with open("config.yaml", "r") as f:
#     data = yaml.safe_load(f)

# print(data)
results = model.train(data="config.yaml", epochs=1) # Training our model

model_path = "runs/detect/"
# model = YOLO("yolov8n.yaml")
# model.load_data("config.yaml") # Loading data
# model.epochs = 1 # Setting the number of epochs
# results = model.train() # Training our model
#  2381 images