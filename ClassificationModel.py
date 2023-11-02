import os
import cv2 as cv
import numpy as np 
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator


data_dir = "/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/ClassificationDataset"


def load_image(image_path, convert):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))
    if convert:
        image = np.expand_dims(image, axis=0)
        image_list = image
    else:
        image = np.expand_dims(image, axis=0)

        datagen = ImageDataGenerator(
            rotation_range=90,  # Random rotation between 0 and 90 degrees
            rescale=1./255      # Normalize pixel values to [0, 1]
        )
        image_list = []

        counter = 0

        for batch in datagen.flow(image, batch_size=1):
            image_list.append(batch[0])
            counter += 1

            if counter == 4:
                break

    return image_list

def parse_name(image_path):

    # print(image_path)
    label = 0
    if image_path.startswith(data_dir + "/F"):
        label = 0
    elif image_path.startswith(data_dir + "/M"):
        label = 1
    elif image_path.startswith(data_dir + "/S"):
        label = 2
    elif image_path.startswith(data_dir + "/V"):
        label = 3

    return [label, label, label, label]
    



images = []
all_labels = []

label_to_index = {0: "Fixed Wing Drone", 1: "Multi-Rotor Drone", 2: "Single Rotor Drone", 3: "Fixed Wing Hybrid VTOL"}  # Define mapping for labels
index_to_label = {"Fixed Wing Drone": 0, "Multi-Rotor Drone": 1, "Single Rotor Drone": 2, "Fixed Wing Hybrid VTOL" : 3} # useful when we test our model on new images

for filename in os.listdir(data_dir):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        image_path = os.path.join(data_dir, filename)
        
        labels = parse_name(image_path)            
        image = load_image(image_path, False)
        images.extend(image)
        all_labels.extend(labels)

print(f"total image is {len(images)}")
images = np.array(images)
print(len(all_labels))
all_labels = np.array(all_labels)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))  
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2))) 
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(4, activation = 'softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# print(images.shape)
# print(labels.shape)

# split into training and validation data into 80 20
X_train, X_val, y_train, y_val = train_test_split(images, all_labels, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))


# base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# for layer in base_model.layers:
#     layer.trainable = False

# x = base_model.output
# x = layers.Flatten()(x)

# # Regression head for bounding box
# bbox_output = layers.Dense(3, activation='sigmoid', name='bbox_output')(x)
# # Classification head for object label
# classification_output = layers.Dense(len(label_to_index), activation='softmax', name='class_output')(x)

# model = models.Model(inputs=base_model.input, outputs=[bbox_output, classification_output])

# model.compile(optimizer='adam',
#               loss={'bbox_output': 'mean_squared_error', 'class_output': 'sparse_categorical_crossentropy'},
#               metrics={'class_output': 'accuracy'})

# # Split the data
# X_train, X_val, y_train_boxes, y_val_boxes, y_train_labels, y_val_labels = train_test_split(
#     images, all_boxes, all_labels, test_size=0.2, random_state=42)

# model.fit(X_train, {'bbox_output': y_train_boxes, 'class_output': y_train_labels},
#           validation_data=(X_val, {'bbox_output': y_val_boxes, 'class_output': y_val_labels}),
#           epochs=1, batch_size=32)



# test_image1 = load_image("/Users/arjavjain/Documents/NGHackWeekTeam4/TRAIN", True)
# test_res1 = model.predict(test_image1)
# # Get the class label index with the highest probability
# class_label_index1 = np.argmax(test_res1[1])
# # Map the index to the class name
# class_name1 = label_to_index[class_label_index1]
# print(f'The predicted class is: {class_name1}')

model.save("/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/Classification") # save the model and so that we can use for later pull out easier


