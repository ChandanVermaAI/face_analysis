import cv2
import platform
import subprocess
import my_loger
logger = my_loger.setup_logger()
try:
    import wmi
except ImportError:
    logger.error("wmi module is only available on Windows.")

#logger.basicConfig(level=logging.INFO)

def get_camera_name():
    try:
        if platform.system() == "Windows":
            c = wmi.WMI()
            for device in c.Win32_PnPEntity():
                if "camera" in device.Name.lower() or "video" in device.Name.lower():
                    return device.Name
            return "Unknown Camera"
        elif platform.system() == "Linux":
            result = subprocess.run(['v4l2-ctl', '--list-devices'], capture_output=True, text=True)
            return result.stdout.split('\n')[0]  # Assuming the camera name is in the first line
        else:
            return "Unsupported OS"
    except Exception as e:
        logger.error(f"An error occurred while getting camera name: {e}")
        return "Unknown Camera"

def check_camera():
    try:
        # Attempt to open the default camera (index 0)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.info("No camera detected.")
            return False, "No camera detected."
        
        # Retrieve camera properties (these properties might vary based on the camera and system)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        camera_name = get_camera_name()

        camera_details = {
            "Camera Name": camera_name,
            "Frame Width": frame_width,
            "Frame Height": frame_height,
            "FPS": fps
        }
        
        logger.info("Camera detected.")
        logger.info(f"Camera details: {camera_details}")
        
        cap.release()
        return True, camera_details
    
    except Exception as e:
        logger.error(f"An error occurred while checking for camera: {e}")
        return False, str(e)
