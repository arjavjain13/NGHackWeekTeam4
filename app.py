from flask import Flask, request, render_template
from testingModels import runTest, show_images_with_boxes, load_image, label_to_index, parse_txt_here
import os
import cv2 as cv

app = Flask(__name__, template_folder='templates')


# Set the upload folder to a temporary location
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get the uploaded image and text files
        uploaded_image = request.files['image']
        uploaded_text = request.files['text']

        if uploaded_image and uploaded_text:
            # Save the uploaded files to the upload folder
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
            text_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_text.filename)
            uploaded_image.save(image_path)
            uploaded_text.save(text_path)

            # Use the uploaded image for testing
            image_result = runTest(image_path)
            image_curr = load_image(image_path, convert=False)
            
            # The path to the modified image within the static folder
            modified_image_path = 'images/modified_image.jpg'
            
            if image_result != "There was no drone detected in the picture":
                l, boxes = parse_txt_here(text_path)
                modified_image = show_images_with_boxes(image_curr, boxes[0])
                modified_image_path = image_path[:3] + "_mod.jpg"
                modified_image.save(modified_image_path)
                # modified_image_path = os.path.join('static', 'images', 'modified_image.jpg')
                # cv.imwrite(modified_image_path, modified_image)

                return render_template('result.html', result=image_result, picture=modified_image_path)
            else:
                return render_template('result.html', result=image_result)


    return "Upload failed"


if __name__ == '__main__':
    app.run(debug=True)
