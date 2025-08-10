from src.data_ingestion import DataIngestion
from src.feature_store import RedisFeatureStore
from src.data_processing import DataProcessing
from src.model_training import ModelTraining

from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIR
from config.paths_config import TRAIN_PATH, TEST_PATH

if __name__ == "__main__":
    # data ingestion pipeline
    data_ingestion_obj = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion_obj.run()

    # data preprocessing
    feature_store = RedisFeatureStore()
    data_processor = DataProcessing(
                        train_data_path= TRAIN_PATH ,
                        test_data_path= TEST_PATH,
                        feature_store=feature_store )
    data_processor.run()

    #model trainining
    feature_store = RedisFeatureStore()
    model_traininig = ModelTraining(feature_store)
    model_traininig.run()



    