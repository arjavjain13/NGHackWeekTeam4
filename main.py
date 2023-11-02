import os
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications

data_dir = "/Users/zhenzhang/Desktop/CS courses /NGHackWeekTeam4/TRAIN_txt" # replace the path

####  we are using the VGG16 model from the tensorflow.keras.applications
def load_image(image_path, convert):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))
    if convert:
        image = np.expand_dims(image, axis=0)
    image = image / 255.0
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

def show_images_with_boxes(images, boxes, labels):
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))  # Adjust the size as needed
    axes = axes.ravel()

    for i in np.arange(0, 20):  # Show the first 20 images
        axes[i].imshow(images[i])
        axes[i].set_title(f'Label: {label_to_index[labels[i]]}', fontsize=12)
        
        # Add bounding box if label is 0 (drone)
        if labels[i] == 0:
            # Convert bounding box from relative coordinates to image coordinates
            height, width, _ = images[i].shape
            center_x, center_y, w, h = boxes[i]
            x = int((center_x * width) - (w * width) / 2)
            y = int((center_y * height) - (h * height) / 2)
            w = int(w * width)
            h = int(h * height)

            # Create a rectangle patch and add it to the axis
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
        
        # Remove axes for cleaner look
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.show()


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

print(f"total image is {total}")
images = np.array(images)
all_labels = np.array(all_labels)
all_boxes = np.array(all_boxes)


# Call the function to display images
show_images_with_boxes(images, all_boxes, all_labels)
# build the model 
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
          epochs=10, batch_size=32)



test_image = load_image("/Users/zhenzhang/Desktop/CS courses /NGHackWeekTeam4/testimage/test.jpg", True)
test_res = model.predict(test_image)
# Get the class label index with the highest probability
class_label_index = np.argmax(test_res[1])
# Map the index to the class name
class_name = label_to_index[class_label_index]

print(f'The predicted class is: {class_name}')

test_image1 = load_image("/Users/zhenzhang/Desktop/CS courses /NGHackWeekTeam4/testimage/test3.jpg", True)
test_res1 = model.predict(test_image1)
# Get the class label index with the highest probability
class_label_index1 = np.argmax(test_res1[1])
# Map the index to the class name
class_name1 = label_to_index[class_label_index1]
print(f'The predicted class is: {class_name1}')

model.save("drone_detection_model/my_model")  # save the model


