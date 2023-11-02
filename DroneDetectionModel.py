import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications

data_dir = "/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/TRAIN_txt"

####  we are using the VGG16 model from the tensorflow.keras.applications
def load_image(image_path, convert):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))
    if convert:
        image = np.expand_dims(image, axis=0)
    image = image / 255.0
    print(image.shape)
    return image

def parse_txt(txt_path):
    # Check if txt file exists
    if not os.path.exists(txt_path):
        return [1], [[0, 0, 0, 0]]  # Return 1 label stand for no drone and dummy coordinates

    with open(txt_path, 'r') as file:
        lines = file.readlines()
        
        first_line = lines[0].strip().split()
        
        label = int(first_line[0])
        box = [float(coord) for coord in first_line[1:]]
        
        return [label], [box]

def show_images_with_boxes(image, box):

    if len(image.shape) == 3:
        height, width, _ = image.shape
        center_x, center_y, w, h = box
        x = int((center_x * width) - (w * width) / 2)
        y = int((center_y * height) - (h * height) / 2)
        w = int(w * width)
        h = int(h * height)

        # Create a rectangle patch and add it to the axis
        modified_image = cv.rectangle(image, (x, y), w, h, (0, 0, 255), 2)

        return modified_image
    else:
        return None


images = []
all_labels = []
all_boxes = []


label_to_index = {0: "drone", 1: "no drone"}  # Define mapping for labels
index_to_label = {"drone": 0, "no drone": 1} # useful when we test our model on new images

total = 0 
for filename in os.listdir(data_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(data_dir, filename)
        txt_path = os.path.join(data_dir, base_name + ".txt")
        
        labels, boxes = parse_txt(txt_path)            
        image = load_image(image_path, False)
        images.append(image)
        all_labels.extend(labels)
        all_boxes.extend(boxes)
        total += 1

images = np.array(images)
all_labels = np.array(all_labels)
all_boxes = np.array(all_boxes)

base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = layers.Flatten()(x)

# Regression head for bounding box
bbox_output = layers.Dense(1, activation='sigmoid', name='bbox_output')(x)
# Classification head for object label
classification_output = layers.Dense(len(label_to_index), activation='softmax', name='class_output')(x)

model = models.Model(inputs=base_model.input, outputs=[bbox_output, classification_output])

model.compile(optimizer='adam',
              loss={'bbox_output': 'mean_squared_error', 'class_output': 'sparse_categorical_crossentropy'},
              metrics={'class_output': 'accuracy'})

# Split the data
X_train, X_val, y_train_boxes, y_val_boxes, y_train_labels, y_val_labels = train_test_split(
    images, all_boxes, all_labels, test_size=0.2, random_state=42)

model.fit(X_train, {'bbox_output': y_train_boxes, 'class_output': y_train_labels},
          validation_data=(X_val, {'bbox_output': y_val_boxes, 'class_output': y_val_labels}),
          epochs=1, batch_size=32)

model.save("/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/DroneDetection")  # save the model
