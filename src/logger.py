import logging
import os
from datetime import datetime

# Create logs directory
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Generate log file path
log_filename = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_file_path = os.path.join(logs_dir, log_filename)

# Configure logging
logging.basicConfig(
    filename = log_file_path,
    format = '[%(asctime)s] %(levelname)s: %(message)s',
    level = logging.INFO,
    datefmt = '%Y-%m-%d %H:%M:%S'
)