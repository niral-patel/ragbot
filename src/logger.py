# src/logger.py : Set up logging for the application. This will help us track the flow of the application and debug any issues that may arise. single logging setup reused across every file in the projects
# anyfile does -> from src.logger import logger
# then use logger.info(), logger.error() etc to log messages

import logging
import os
from datetime import datetime
import sys

# --step - 1: Build log file name ---
# Every time you run the project, a NEW log file is created with a timestamp
# Example filename: 03-10-2026-14-30-22.log
# This matches exactly what you saw in your image (logs/ folder)

LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"

# --step - 2: Create logs directory if it doesn't exist ---
# os.getcwd() = the folder you run the script FROM (always project root)
# exist_ok=True = don't crash if folder already exists

LOG_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# --step - 3: Configure logging ---
# level=logging.INFO = log INFO and above (INFO, WARNING, ERROR, CRITICAL)
# format = how the log messages will look like in the file
# datefmt = how the timestamp will look like in the log messages
# We get a named logger (not the root logger)
# __name__ = "src.logger" — helps identify WHERE a log message came from

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --step - 4: Create file handler to write logs to the file ---
# Every log line will look like:
# [2026-03-10 14:30:22] 42 src.logger - INFO - Your message here
#  ↑ timestamp          ↑ line number  ↑ file     ↑ level  ↑ message
if not logger.hasHandlers():

    formatter = logging.Formatter(
        '[%(asctime)s] Line:%(lineno)d "ragbot" File:%(name)s - %(levelname)s - %(message)s', 
        datefmt='%m-%d-%Y %H:%M:%S'
        )

    
    #--step - 5: Terminal handler (see logs live while coding) ---
    # This will print logs to the terminal as well as the file
    console_handler = logging.StreamHandler(
        stream=open(sys.stdout.fileno(), 
                    mode='w', 
                    encoding='utf-8', 
                    buffering=1)
    )
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    #--step - 6: File handler (saves logs to disk permanently) ---
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    #--step - 7: Add handlers to the logger ---
    #Guard prevents adding multiple handlers if this file is imported multiple times
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)