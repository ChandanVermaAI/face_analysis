import os
import cv2
import numpy as np
from keras.preprocessing import image as keras_image
from keras.models import model_from_json, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, AveragePooling2D
import face_recognition
import tensorflow as tf

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Function to analyze emotions in an image
def analyze_emotions(image_path):
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version: {tf.keras.__version__}")
    
    # Loading the image to detect
    image_to_detect = cv2.imread(image_path)
    if image_to_detect is None:
        print(f"Error: Cannot read the image at {image_path}")
        return
    
    # Load the model and load the weights
    with open("dataset/facial_expression_model_structure.json", "r") as json_file:
        model_json = json_file.read()
    
    try:
        face_exp_model = model_from_json(model_json)
        face_exp_model.load_weights('dataset/facial_expression_model_weights.h5')
    except Exception as e:
        print(f"Error loading the model: {e}")
        return
    
    # Declare the emotions label
    emotions_label = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    
    # Detect all faces in the image
    all_face_locations = face_recognition.face_locations(image_to_detect, model='hog')
    
    # Print the number of faces detected
    print(f'There are {len(all_face_locations)} faces in this image')
    
    # Looping through the face locations
    for index, current_face_location in enumerate(all_face_locations):
        # Splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Printing the location of the current face
        print(f'Found face {index + 1} at top: {top_pos}, right: {right_pos}, bottom: {bottom_pos}, left: {left_pos}')
        
        # Slicing the current face from the main image
        current_face_image = image_to_detect[top_pos:bottom_pos, left_pos:right_pos]
        
        # Draw rectangle around the face detected
        cv2.rectangle(image_to_detect, (left_pos, top_pos), (right_pos, bottom_pos), (0, 0, 255), 2)
        
        # Preprocess input, convert it to an image like as the data in the dataset
        # Convert to grayscale
        current_face_image = cv2.cvtColor(current_face_image, cv2.COLOR_BGR2GRAY)
        # Resize to 48x48 px size
        current_face_image = cv2.resize(current_face_image, (48, 48))
        # Convert the image into a 3D numpy array
        img_pixels = keras_image.img_to_array(current_face_image)
        # Expand the shape of the array into a single row with multiple columns
        img_pixels = np.expand_dims(img_pixels, axis=0)
        # Normalize all pixels in the range [0, 1]
        img_pixels /= 255
        
        # Predict using the model, get the prediction values for all 7 expressions
        exp_predictions = face_exp_model.predict(img_pixels)
        # Find max indexed prediction value (0 to 6)
        max_index = np.argmax(exp_predictions[0])
        # Get corresponding label from emotions_label
        emotion_label = emotions_label[max_index]
        
        # Display the name as text in the image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image_to_detect, emotion_label, (left_pos, bottom_pos + 20), font, 0.5, (255, 255, 255), 1)
    
    # Show the image with rectangles and labels
    cv2.imshow("Image Face Emotions", image_to_detect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'dataset/testing/joe1.jpg'
analyze_emotions(image_path)
