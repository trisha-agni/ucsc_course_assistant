# External Imports
from datetime import datetime
import logging
import os
import sys
# Internal Imports

LOGS_DIR_NAME = 'logs'


def _setup_logger_levels(logger):
    logger.setLevel(logging.INFO)
    # --- SILENCE THIRD-PARTY JUNK LOGS ---
    # Mute noisy HTTP request logging from OpenRouter/LangChain API calls
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # Mute local CPU/GPU hardware notices from the embedding models
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    # Completely suppress unauthenticated HF_TOKEN download warnings
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def setup_logging():
    """Configure centralized logging for the entire app"""
    logger = logging.getLogger('')
    # if handlers already exist, don't re-add them (prevents streamlit
    # duplicate logs)
    if logger.handlers:
        return logger
    _setup_logger_levels(logger)
    logger.setLevel(logging.INFO)

    log_format = "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # resolve paths
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(cur_dir, '..', '..', '..'))
    # path to the logs dir
    logs_dir = os.path.join(root_dir, LOGS_DIR_NAME)
    os.makedirs(logs_dir, exist_ok=True)
    # generate timestamp str for the log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_log_name = f'ucsc_course_assistant.{timestamp}.log'
    log_file_path = os.path.join(logs_dir, dynamic_log_name)
    # file output
    file_handler = logging.FileHandler(
        log_file_path,
        encoding='utf-8',
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
