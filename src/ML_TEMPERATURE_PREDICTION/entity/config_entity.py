from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    batch_size: int = 8
    num_workers: int = 2
    channels: int = 3  # 3 for RGB, 1 for thermal
    # New parameters
    data_augmentation: bool = True
    normalize_mean: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    normalize_std: Tuple[float, float, float] = (0.5, 0.5, 0.5)

@dataclass
class ModelTrainerConfig:
    """
    Configuration for model training
    """
    root_dir: Path
    data_path: Path
    # Basic training parameters
    num_epochs: int = 500
    lr: float = 0.0002
    batch_size: int = 8
    direction: str = 'rgb2thermal'  # 'rgb2thermal' or 'thermal2rgb'
    image_size: int = 256
    
    # Learning rate scheduler
    lr_scheduler: str = "step"
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    warmup_epochs: int = 10
    
    # Loss function weights
    lambda_pixel: float = 200
    lambda_perceptual: float = 10
    lambda_gan: float = 1
    
    # Training strategy
    early_stopping: bool = True
    patience: int = 50
    save_frequency: int = 20
    
    # Model architecture
    generator_filters: int = 64
    discriminator_filters: int = 64
    dropout_rate: float = 0.5
    use_attention: bool = False
    
    # Optimizer settings
    optimizer: str = "Adam"
    beta1: float = 0.5
    beta2: float = 0.999
    weight_decay: float = 1e-4
    
    # Mixed precision
    use_amp: bool = False

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
    batch_size: int = 8

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