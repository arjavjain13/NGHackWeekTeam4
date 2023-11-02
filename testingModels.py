from tensorflow.keras.models import load_model # import the model 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2 as cv
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

label_to_index = {0: "Fixed Wing Drone", 1: "Multi-Rotor Drone", 2: "Single Rotor Drone", 3: "Fixed Wing Hybrid VTOL"}

def show_images_with_boxes(image, box):
    # Clone the original image to preserve it
    modified_image = image.copy()

    # Convert bounding box from relative coordinates to image coordinates
    height, width, _ = image.shape
    center_x, center_y, w, h = box
    x = int((center_x * width) - (w * width) / 2)
    y = int((center_y * height) - (h * height) / 2)
    w = int(w * width)
    h = int(h * height)

    # Create a red outline around the bounding box
    cv.rectangle(modified_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Using (0, 0, 255) for red color

    return modified_image



def load_image(image_path, convert):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))
    if convert:
        image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def parse_txt_here(txt_path):
    # Check if txt file exists
    if not os.path.exists(txt_path):
        return [1], [[0, 0, 0, 0]]  # Return 1 label stand for no drone and dummy coordinates

    with open(txt_path, 'r') as file:
        lines = file.readlines()
        
        first_line = lines[0].strip().split()
        
        label = int(first_line[0])
        box = [float(coord) for coord in first_line[1:]]
        
        return [label], [box]

    
def runTest(test_image_path):
    detection_model = load_model("/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/DroneDetection", compile=False)
    test_image = load_image(test_image_path, convert=True)
    detected = detection_model.predict(test_image)

    if np.argmax(detected[0], axis=1) == 0:
        classification_model = load_model("/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/Classification", compile=False)
        prediction = classification_model.predict(test_image)
        class_label_index = np.argmax(prediction, axis=1)
        class_name = label_to_index[class_label_index[0]]
        return f'A drone was detected in the image, and the predicted class is: {class_name}'
    else:
        return 'There was no drone detected in the picture'
