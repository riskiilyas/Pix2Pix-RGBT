from ML_TEMPERATURE_PREDICTION.components.model_evaluation import ModelEvaluation
from ML_TEMPERATURE_PREDICTION.config.configuration import ConfigurationManager
from ML_TEMPERATURE_PREDICTION.logging import logger

class PredictionPipeline:
    def __init__(self):
        pass
    
    def main(self, input_type=None):
        try:
            config = ConfigurationManager()
            
            # Get prediction config
            prediction_config = config.get_prediction_config()
            
            # Get evaluation config for model path
            model_evaluation_config = config.get_model_evaluation_config()
            
            # Use ModelEvaluation for prediction since it has the generate_predictions method
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            
            # If input_type is not specified, infer it from the direction
            if input_type is None:
                input_type = 'rgb' if prediction_config.direction == 'rgb2thermal' else 'thermal'
            
            # Generate predictions
            model_evaluation.generate_predictions(
                input_dir=prediction_config.input_dir,
                output_dir=prediction_config.output_dir,
                input_type=input_type
            )
            
            logger.info(f"Predictions generated from {prediction_config.input_dir} to {prediction_config.output_dir}")
            
        except Exception as e:
            logger.exception(e)
            raise e

if __name__ == "__main__":
    try:
        logger.info("Prediction pipeline started")
        obj = PredictionPipeline()
        obj.main()
        logger.info("Prediction pipeline completed")
    except Exception as e:
        logger.exception(e)
        raise e