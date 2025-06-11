import os
from pathlib import Path
from src.ML_TEMPERATURE_PREDICTION.utils.common import read_yaml, create_directories
from src.ML_TEMPERATURE_PREDICTION.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    PredictionConfig
)
from src.ML_TEMPERATURE_PREDICTION.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH

class ConfigurationManager:
    """
    Configuration manager to create various configuration objects
    """
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Create artifacts directory
        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get configuration for data ingestion
        """
        config = self.config.data_ingestion
        
        create_directories([Path(config.root_dir)])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            local_data_file=Path(config.local_data_file),
            train_ratio=self.params.TRAIN_RATIO,
            val_ratio=self.params.VAL_RATIO,
            test_ratio=self.params.TEST_RATIO,
            random_state=self.params.RANDOM_STATE
        )
        
        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Get configuration for data transformation
        """
        config = self.config.data_transformation
        
        create_directories([Path(config.root_dir)])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            image_size=self.params.IMAGE_SIZE,
            batch_size=self.params.BATCH_SIZE,
            num_workers=self.params.NUM_WORKERS,
            channels=3 if self.params.DIRECTION == 'thermal2rgb' else 1,
            # New parameters
            data_augmentation=self.params.DATA_AUGMENTATION,
            normalize_mean=tuple(self.params.NORMALIZE_MEAN),
            normalize_std=tuple(self.params.NORMALIZE_STD)
        )
        
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Get configuration for model training
        """
        config = self.config.model_trainer
        
        create_directories([Path(config.root_dir)])
        
        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            # Basic parameters
            num_epochs=self.params.NUM_EPOCHS,
            lr=self.params.LEARNING_RATE,
            batch_size=self.params.BATCH_SIZE,
            direction=self.params.DIRECTION,
            image_size=self.params.IMAGE_SIZE,
            # Learning rate scheduler
            lr_scheduler=self.params.LR_SCHEDULER,
            lr_step_size=self.params.LR_STEP_SIZE,
            lr_gamma=self.params.LR_GAMMA,
            warmup_epochs=self.params.WARMUP_EPOCHS,
            # Loss weights
            lambda_pixel=self.params.LAMBDA_PIXEL,
            lambda_perceptual=self.params.LAMBDA_PERCEPTUAL,
            lambda_gan=self.params.LAMBDA_GAN,
            # Training strategy
            early_stopping=self.params.EARLY_STOPPING,
            patience=self.params.PATIENCE,
            save_frequency=self.params.SAVE_FREQUENCY,
            # Model architecture
            generator_filters=self.params.GENERATOR_FILTERS,
            discriminator_filters=self.params.DISCRIMINATOR_FILTERS,
            dropout_rate=self.params.DROPOUT_RATE,
            use_attention=self.params.USE_ATTENTION,
            # Optimizer
            optimizer=self.params.OPTIMIZER,
            beta1=self.params.BETA1,
            beta2=self.params.BETA2,
            weight_decay=self.params.WEIGHT_DECAY,
            # Mixed precision
            use_amp=self.params.USE_AMP
        )
        
        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Get configuration for model evaluation
        """
        config = self.config.model_evaluation
        
        create_directories([Path(config.root_dir)])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            evaluation_path=Path(config.evaluation_path),
            image_size=self.params.IMAGE_SIZE,
            direction=self.params.DIRECTION,
            batch_size=self.params.BATCH_SIZE
        )
        
        return model_evaluation_config
    
    def get_prediction_config(self) -> PredictionConfig:
        """
        Get configuration for generating predictions
        """
        config = self.config.prediction
        
        create_directories([Path(config.output_dir)])
        
        prediction_config = PredictionConfig(
            model_path=Path(config.model_path),
            input_dir=Path(config.input_dir),
            output_dir=Path(config.output_dir),
            direction=self.params.DIRECTION,
            image_size=self.params.IMAGE_SIZE
        )
        
        return prediction_config