import os
import shutil
from pathlib import Path
from src.ML_TEMPERATURE_PREDICTION.entity.config_entity import DataIngestionConfig
from src.ML_TEMPERATURE_PREDICTION.logging import logger
import splitfolders

class DataIngestion:
    """
    Data ingestion component for processing RGB and thermal images
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        The data should be downloaded or copied from local repository
        If the data directory already exists, it will be skipped
        """
        if not os.path.exists(self.config.local_data_file):
            logger.info(f"Data directory not found at {self.config.local_data_file}")
            logger.info("Please ensure data directory exists with 'rgb' and 'thermal' subdirectories")
            raise Exception(f"Data directory not found at {self.config.local_data_file}")
        
        # Check if rgb and thermal subdirectories exist
        rgb_dir = os.path.join(self.config.local_data_file, 'rgb')
        thermal_dir = os.path.join(self.config.local_data_file, 'thermal')
        
        if not os.path.exists(rgb_dir) or not os.path.exists(thermal_dir):
            logger.info(f"RGB or Thermal subdirectories not found at {self.config.local_data_file}")
            raise Exception(f"RGB or Thermal subdirectories not found at {self.config.local_data_file}")
        
        logger.info(f"Found data directory at {self.config.local_data_file}")
        
    def extract_zip_file(self):
        """
        Not needed as we're using local directories directly
        """
        pass

    def split_data(self):
        """
        Split the data into train, validation, and test sets
        """
        # Create output directory if it doesn't exist
        os.makedirs(self.config.data_path, exist_ok=True)
        
        # Split data into train, validation, test
        logger.info(f"Splitting data into train, validation, and test sets")
        splitfolders.ratio(
            self.config.local_data_file,
            output=self.config.data_path,
            seed=self.config.random_state,
            ratio=(self.config.train_ratio, self.config.val_ratio, self.config.test_ratio),
            group_prefix=None
        )
        
        logger.info(f"Data split complete")