import streamlit as st
import pandas as pd
import camera_test
import face_detection_helper
import tempfile
import emotion_analysis_function
import my_loger
import cv2
from PIL import Image
from deepface import DeepFace

# Set up logging
logger = my_loger.setup_logger()
st.set_page_config(
    page_title="Emotion Detection Tool",
    page_icon=":emotion:",
    layout="centered",  # Center the content
)

st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #86D8DB;
    }
    .stApp {
        background-color: #73D9C8;  /* Background color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define title with emoji and image
st.markdown(
    """
    <h2 style="font-size:24px; color: #4CAF50;">
        👤 System for Human Face Detection and Analysis
    </h2>
    """,
    unsafe_allow_html=True
)
st.markdown("Choose an input video or upload an image to analyze emotions.")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Button to trigger image analysis
if uploaded_image is not None:
    st.image(uploaded_image, use_column_width=True, caption="Uploaded Image")
    image = Image.open(uploaded_image)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_image.getvalue())
        temp_file_path = temp_file.name

    #print(image)
    c1, c2 = st.columns(2)
    r1, r2 = emotion_analysis_function.analyze_image(temp_file_path)
    r1 = pd.DataFrame(r1)
    print(r1)

    if c1.button("Analysis Result", key="analyze_result"):
        if r2 is not None:
            st.dataframe(r2)
        else:
            st.error("No face detected in the image.")

    c2.download_button(
        label="Download Analysis Results",
        data=r1.to_csv(index=False),
        file_name=temp_file.name + 'analysis_results.csv',
        mime='text/csv',
        key="download_analysis"
    )

st.markdown(' ')
st.empty()
st.markdown('<h3 style="color:blue;">Real-Time Facial Analysis</h2>', unsafe_allow_html=True)
st.markdown('<h5 style="color:green;">Camera Testing: Test Your System Camera</h2>', unsafe_allow_html=True)

# Function to analyze a frame using DeepFace
def analyze_frame(frame):
    try:
        # Analyze the frame
        results = DeepFace.analyze(
            img_path=frame,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False  # Set to False to avoid stopping if no face is detected
        )
        return results
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return None

# Function to start real-time facial analysis using OpenCV and DeepFace
def start_real_time_analysis():
    cap = cv2.VideoCapture(0)
    st_frame = st.empty()

    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        # Convert the frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame
        results = analyze_frame(rgb_frame)

        if results:
            age = results[0]['age']
            gender = results[0]['dominant_gender']
            race = results[0]['dominant_race']
            emotion = results[0]['dominant_emotion']
            # Display the results on the frame
            cv2.putText(frame, f"Age: {age}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Race: {race}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame back to RGB for Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(rgb_frame, channels="RGB")

        if st.button("Stop", key="stop_analysis"):
            st.info("stoping")
            break

    cap.release()
    cv2.destroyAllWindows()

if 'camera_detected' not in st.session_state:
    st.session_state.camera_detected = False

if st.button("Camera Testing", key="camera_testing"):
    try:
        camera_detected, camera_info = camera_test.check_camera()
        if camera_detected:
            st.session_state.camera_detected = True
            st.success("Camera detected!")
            with st.expander("Camera Details"):
                st.write(camera_info)
        else:
            st.session_state.camera_detected = False
            st.error(camera_info)
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logger.error(f"An unexpected error occurred: {e}")

if st.session_state.camera_detected:
    # Open a new section with "Detect Face" button
    st.markdown(
        """
        <h3 style="color: #4CAF50; font-size:22px;">
            Face Detection Section
        </h3>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("Open System Camera", key="open_camera"):
        logger.info("Running face analysis function")
        face_detect_result = face_detection_helper.detect_face_in_video()
        if face_detect_result:
            st.success("Face detected!")
            st.info("Running face analysis function")
            start_real_time_analysis()
        else:
            st.error("No face detected.")

# Sidebar with styled header and radio buttons for navigation
st.sidebar.markdown(
    """
    <h3 style="color: #FFA500; font-size:20px;">
        Navigation
    </h3>
    """,
    unsafe_allow_html=True
)

navigation = st.sidebar.radio("Go to", ["About Me", "Team", "Project Details", "Code Details"])

# Display content based on sidebar radio selection
if navigation == "About Me":
    st.header("About Me")
    st.write("Details about me.")

elif navigation == "Team":
    st.header("Team")
    st.write("Details about the team.")

elif navigation == "Project Details":
    st.header("Project Details")
    st.write("Details about the project.")

elif navigation == "Code Details":
    st.header("Code Details")
    st.write("Details about the code.")
