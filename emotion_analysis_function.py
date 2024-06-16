
from deepface import DeepFace
import cv2
import os
import my_loger
logger=my_loger.setup_logger()
# Function to analyze an image using DeepFace
def analyze_image(image_path):
    try:
        
        results = DeepFace.analyze(img_path=image_path)    
        logger.info("Image analysis results completed")
        dominant_emotion = results[0]['dominant_emotion']
        dominant_race = results[0]['dominant_race']
        dominant_gender = results[0]['dominant_gender']
        age = results[0]['age']
        dict_result={"age":age,"gender":dominant_gender,"emotion":dominant_emotion,"race":dominant_race}


        return results,dict_result

    except Exception as e:
        # Log the exception
        logger.error("Error analyzing image: %s", str(e))
        print(f"Error analyzing image: {e}")

def start_deepface_stream():
    try:
        DeepFace.stream(
            #db_path=None,
            model_name='VGG-Face', # You can choose other models like 'Facenet', 'OpenFace', etc.
            detector_backend='opencv',
            distance_metric='cosine',
            enable_face_analysis=True,
            source=0, # 0 for webcam
            time_threshold=5, # Time threshold for recognition (seconds)
            frame_threshold=5, # Frame threshold for recognition
        )
    except Exception as e:
        logger.error(f"Error starting DeepFace stream: {e}")


start_deepface_stream()


