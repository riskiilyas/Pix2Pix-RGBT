from ML_TEMPERATURE_PREDICTION.components.data_ingestion import DataIngestion
from ML_TEMPERATURE_PREDICTION.config.configuration import ConfigurationManager
from ML_TEMPERATURE_PREDICTION.logging import logger

class DataIngestionPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.split_data()

if __name__ == "__main__":
    try:
        logger.info("Data ingestion pipeline started")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info("Data ingestion pipeline completed")
    except Exception as e:
        logger.exception(e)
        raise e