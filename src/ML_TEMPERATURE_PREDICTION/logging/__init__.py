import os
import sys
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create log file with timestamp
log_file = os.path.join(logs_dir, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log")

# Logging configuration
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger("temperature_prediction_logger")