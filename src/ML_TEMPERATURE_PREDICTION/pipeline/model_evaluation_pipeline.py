from ML_TEMPERATURE_PREDICTION.components.data_transformation import DataTransformation
from ML_TEMPERATURE_PREDICTION.components.model_evaluation import ModelEvaluation
from ML_TEMPERATURE_PREDICTION.config.configuration import ConfigurationManager
from ML_TEMPERATURE_PREDICTION.logging import logger

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config = ConfigurationManager()
            
            # Data transformation to get test loader
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            _, _, test_loader = data_transformation.get_data_loaders()
            
            # Model evaluation
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            metrics = model_evaluation.evaluate(test_loader)
            
            return metrics
            
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logger.info("Model evaluation pipeline started")
        obj = ModelEvaluationPipeline()
        metrics = obj.main()
        logger.info(f"Model evaluation pipeline completed with metrics: {metrics}")
    except Exception as e:
        logger.exception(e)
        raise e