
import os
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from xml.etree import ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models  


# Directory containing images and XML files
data_dir = "/Users/arjavjain/Desktop/NGHackWeek/TRAIN_xml_format"

# Load and preprocess an image
def load_image(image_path, convert):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
    image = cv.resize(image, (224, 224))
    print(image_path, len(image.shape) == 3, image.shape)

    if convert:
        image = np.expand_dims(image, axis=0)
        # print(image_path)
        # print("After", image.shape)
    image = image / 255.0  # Normalize to [0, 1]
    return image

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    labels = {}
    for object_tag in root.findall('object'):
        label = object_tag.find('name').text
        xmin = float(object_tag.find('bndbox/xmin').text)
        ymin = float(object_tag.find('bndbox/ymin').text)
        xmax = float(object_tag.find('bndbox/xmax').text)
        ymax = float(object_tag.find('bndbox/ymax').text)
        # depth = float(object_tag.find('depth').text)
        # height = float(object_tag.find('height').text)
        # width = float(object_tag.find('width').text)

        # You can preprocess and format these values as needed
        # For example, normalize the bounding box coordinates and scale the depth, height, and width

        labels.update({
            "label": label,
            "bbox": (xmin, ymin, xmax, ymax),
            # "depth": depth,
            # "height": height,
            # "width": width
        })

    return labels

# Load dataset
images = []
labels_label = []
# label_bbox = []
label_to_index = {}  # Dictionary to convert labels to integer indices
index = 0

# extract the images and label from the folder and them put them in array 
for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(data_dir, filename)
        xml_path = os.path.join(data_dir, base_name + ".xml")

        if os.path.exists(xml_path):
            image_info = parse_xml(xml_path)

            # if label.get("label") not in label_to_index:
            #     label_to_index[label.get("index")] = index
            #     index += 1
            # label_index = label_to_index[label.get("index")]

            image = load_image(image_path, False)
            images.append(image)
            labels_label.append(image_info.get("label"))
            

images = np.array(images)
labels = np.array(labels_label)

# print(labels)

# print(labels)

# define the CNN 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# split into training and validation data into 80 20
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.fit_transform(y_val)
# Train model
model.fit(X_train, y_train_encoded, epochs=20, validation_data=(X_val, y_val_encoded))
# Train model; validation_ data are the test data we need ask to get from mentor. 
#model.fit(images, labels, epochs=10, validation_data = (test_image,test_label))


# Test model
test_image = load_image("/Users/arjavjain/Desktop/NGHackWeek/TEST_xml_format/pic_942.jpg", True)
test_res = model.predict(test_image)
print(test_res)
res = np.argmax(test_res, axis= 1)
# print(res)



