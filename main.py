
import os
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models  


# Directory containing images and XML files
data_dir = "."

# Load and preprocess an image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize to [0, 1]
    return image

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = []
    for object_tag in root.findall('object'):
        label = object_tag.find('name').text
        xmin = float(object_tag.find('bndbox/xmin').text)
        ymin = float(object_tag.find('bndbox/ymin').text)
        xmax = float(object_tag.find('bndbox/xmax').text)
        ymax = float(object_tag.find('bndbox/ymax').text)
        depth = float(object_tag.find('size/depth').text)
        height = float(object_tag.find('size/height').text)
        width = float(object_tag.find('size/width').text)

        # You can preprocess and format these values as needed
        # For example, normalize the bounding box coordinates and scale the depth, height, and width

        labels.append({
            "label": label,
            "bbox": (xmin, ymin, xmax, ymax),
            "depth": depth,
            "height": height,
            "width": width
        })

    return labels

# Load dataset
images = []
labels = []
label_to_index = {}  # Dictionary to convert labels to integer indices
index = 0

# extract the images and label from the folder and them put them in array 
for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(data_dir, filename)
        xml_path = os.path.join(data_dir, base_name + ".xml")

        if os.path.exists(xml_path):
            label = parse_xml(xml_path)
            if label not in label_to_index:
                label_to_index[label] = index
                index += 1
            label_index = label_to_index[label]

            image = load_image(image_path)
            images.append(image)
            labels.append(label_index)

images = np.array(images)
labels = to_categorical(np.array(labels))

# define the CNN 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Flattern())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# split into training and validation data into 80 20
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
# Train model; validation_ data are the test data we need ask to get from mentor. 
#model.fit(images, labels, epochs=10, validation_data = (test_image,test_label))


