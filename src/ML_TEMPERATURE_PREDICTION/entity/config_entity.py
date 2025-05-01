from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion
    """
    root_dir: Path
    data_path: Path
    local_data_file: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation
    """
    root_dir: Path
    data_path: Path
    image_size: int = 256
    batch_size: int = 16
    num_workers: int = 4
    channels: int = 3  # 3 for RGB, 1 for thermal

@dataclass
class ModelTrainerConfig:
    """
    Configuration for model training
    """
    root_dir: Path
    data_path: Path
    num_epochs: int = 200
    lr: float = 0.0002
    batch_size: int = 16
    direction: str = 'rgb2thermal'  # 'rgb2thermal' or 'thermal2rgb'
    image_size: int = 256

@dataclass
class ModelEvaluationConfig:
    """
    Configuration for model evaluation
    """
    root_dir: Path
    model_path: Path
    evaluation_path: Path
    image_size: int = 256
    direction: str = 'rgb2thermal'  # 'rgb2thermal' or 'thermal2rgb'

@dataclass
class PredictionConfig:
    """
    Configuration for generating predictions
    """
    model_path: Path
    input_dir: Path
    output_dir: Path
    direction: str = 'rgb2thermal'  # 'rgb2thermal' or 'thermal2rgb'
    image_size: int = 256