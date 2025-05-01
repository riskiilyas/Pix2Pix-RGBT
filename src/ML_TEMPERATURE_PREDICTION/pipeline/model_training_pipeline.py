from src.ML_TEMPERATURE_PREDICTION.components.data_transformation import DataTransformation
from src.ML_TEMPERATURE_PREDICTION.components.model_trainer import ModelTrainer
from src.ML_TEMPERATURE_PREDICTION.config.configuration import ConfigurationManager
from src.ML_TEMPERATURE_PREDICTION.logging import logger

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            
            # Data transformation
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            train_loader, val_loader, test_loader = data_transformation.get_data_loaders()
            
            # Model training
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train(train_loader, val_loader)
            
            return train_loader, val_loader, test_loader
            
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logger.info("Model training pipeline started")
        obj = ModelTrainingPipeline()
        train_loader, val_loader, test_loader = obj.main()
        logger.info("Model training pipeline completed")
    except Exception as e:
        logger.exception(e)
        raise e