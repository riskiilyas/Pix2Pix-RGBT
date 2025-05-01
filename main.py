import os
import argparse
from src.ML_TEMPERATURE_PREDICTION.pipeline.data_ingestion_pipeline import DataIngestionPipeline
from src.ML_TEMPERATURE_PREDICTION.pipeline.model_training_pipeline import ModelTrainingPipeline
from src.ML_TEMPERATURE_PREDICTION.pipeline.model_evaluation_pipeline import ModelEvaluationPipeline
from src.ML_TEMPERATURE_PREDICTION.pipeline.prediction_pipeline import PredictionPipeline
from src.ML_TEMPERATURE_PREDICTION.logging import logger

def run_data_ingestion():
    """Run the data ingestion pipeline"""
    logger.info("Running data ingestion pipeline")
    data_ingestion = DataIngestionPipeline()
    data_ingestion.main()
    logger.info("Data ingestion pipeline completed")

def run_model_training():
    """Run the model training pipeline"""
    logger.info("Running model training pipeline")
    model_training = ModelTrainingPipeline()
    model_training.main()
    logger.info("Model training pipeline completed")

def run_model_evaluation():
    """Run the model evaluation pipeline"""
    logger.info("Running model evaluation pipeline")
    model_evaluation = ModelEvaluationPipeline()
    metrics = model_evaluation.main()
    logger.info(f"Model evaluation pipeline completed with metrics: {metrics}")
    return metrics

def run_prediction(input_type=None):
    """Run the prediction pipeline"""
    logger.info("Running prediction pipeline")
    prediction = PredictionPipeline()
    prediction.main(input_type=input_type)
    logger.info("Prediction pipeline completed")

def run_all():
    """Run all pipelines in sequence"""
    run_data_ingestion()
    run_model_training()
    run_model_evaluation()
    run_prediction()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Temperature Prediction Model")
    
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["data_ingestion", "model_training", "model_evaluation", "prediction", "all"],
        help="Stage of the pipeline to run"
    )
    
    parser.add_argument(
        "--input_type",
        type=str,
        default=None,
        choices=["rgb", "thermal"],
        help="Type of input for prediction (rgb or thermal)"
    )
    
    args = parser.parse_args()
    
    if args.stage == "data_ingestion":
        run_data_ingestion()
    elif args.stage == "model_training":
        run_model_training()
    elif args.stage == "model_evaluation":
        run_model_evaluation()
    elif args.stage == "prediction":
        run_prediction(input_type=args.input_type)
    elif args.stage == "all":
        run_all()