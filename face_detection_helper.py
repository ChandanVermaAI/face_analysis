import cv2
import face_recognition
import my_loger
logger = my_loger.setup_logger()

def detect_face_in_video():
    try:
        # Set up logging
        #logger.basicConfig(level=logger.INFO)
        
        # Capture the video from the default camera
        webcam_video_stream = cv2.VideoCapture(0)

        if not webcam_video_stream.isOpened():
            logger.error("Could not open webcam.")
            return False

        # Initialize the array variable to hold all face locations in the frame
        all_face_locations = []

        # Loop through every frame in the video
        while True:
            # Get the current frame from the video stream as an image
            ret, current_frame = webcam_video_stream.read()

            if not ret:
                logger.error("Failed to grab frame.")
                break

            # Resize the current frame to 1/4 size to process faster
            current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)

            # Detect all faces in the image
            all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=2, model='hog')

            # Check if faces are detected
            face_detected = bool(all_face_locations)
            
            # Log the detection status
            if face_detected:
                logger.info(f"Faces detected: {len(all_face_locations)}")
                return True
            else:
                logger.info("No faces detected.")

            # Break the loop if needed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the stream and camera
        webcam_video_stream.release()
        cv2.destroyAllWindows()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return False

    return False

# Example of using the function
#face_detected = detect_face_in_video()
#print(face_detected)
#logger.info(f"Face detected: {face_detected}")
