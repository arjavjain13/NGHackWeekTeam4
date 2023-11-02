from tensorflow.keras.models import load_model # import the model 
# from ClassificationModel import label_to_index
# from DroneDetectionModel import show_images_with_boxes, parse_txt, load_image
import numpy as np 
import cv2 as cv
import os


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

def show_images_with_boxes(image, box):

    # Convert bounding box from relative coordinates to image coordinates
    height, width, _ = image.shape
    center_x, center_y, w, h = box
    x = int((center_x * width) - (w * width) / 2)
    y = int((center_y * height) - (h * height) / 2)
    w = int(w * width)
    h = int(h * height)

    

    # Create a rectangle patch and add it to the axis
    modified_image = cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return modified_image

test_image = load_image("/Users/arjavjain/Desktop/NGHackWeek/TEST_txt/yoto05134.jpg", False)
l, b = parse_txt("/Users/arjavjain/Desktop/NGHackWeek/TEST_txt/yoto05134.txt")

image = show_images_with_boxes(test_image, b[0])
cv.imshow("Detected Drone", image)      # Display the modified image
cv.waitKey(0)  # Wait for a key event
cv.destroyAllWindows()

# detection_model = load_model("/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/DroneDetection", compile=False)
# detected = detection_model.predict(test_image)
# print("Detection model results:", detected)
# detected = np.argmax(detected, axis=1)
# if detected == 0:
# print("A drone was detected in the image")
# classification_model = load_model("/Users/arjavjain/Documents/GitHub/NGHackWeekTeam4/Classification", compile=False) # use on test new data
# prediction = classification_model.predict(test_image) # do the prediction for new data
# print("Classification model results:", prediction)
# # Get the class label index with the highest probability
# class_label_index = np.argmax(prediction, axis=1)
# # Map the index to the class name
# class_name = label_to_index[class_label_index[0]]

#     print(f'The predicted class is: {class_name}')
    
# else:
#     print("There was no drone detected in the picture")