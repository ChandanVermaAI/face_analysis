import logging
import os
from datetime import datetime
def setup_logger():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Get current date
    current_date = datetime.now().strftime("%d-%m-%Y")

    # Define the log file path
    log_file = os.path.join('logs', f'{current_date}.log')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger()
    return logger


