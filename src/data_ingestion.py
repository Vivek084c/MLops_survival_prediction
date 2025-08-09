import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

from src.logger import get_logger
from src.custom_exception import CustomException

from config.database_config import DB_CONFIG
from config.paths_config import RAW_DIR, TRAIN_PATH, TEST_PATH


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, db_params, output_dir):
        self.db_params = db_params
        self.output_dir = output_dir
    
        os.makedirs(self.output_dir, exist_ok=True)

    def connect_to_db(self):
        try:
            conn = psycopg2.connect(
                host = self.db_params["host"],
                port = self.db_params["port"],
                dbname = self.db_params['dbname'],
                user = self.db_params['user'],
                password = self.db_params['password']
            )

            logger.info("Database connection established")

            return conn
        except Exception as e:
            logger.error(f"Error while establishing the connection {e}")
            raise CustomException(str(e), sys)
    
    def extract_data(self):
        try:
            conn = self.connect_to_db()

            query = "SELECT * FROM public.titanic"
            df = pd.read_sql_query(
                                    sql = query, 
                                    con = conn
                                )   
            conn.close()

            logger.info("Data Extraction from Database compleated")
            return df
        except Exception as e:
            logger.error(f"Error in Data Extraction from database {e}")
            raise CustomException(str(e), sys)
    
    def save_data(self, df):
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(TRAIN_PATH, index = False)
            test_df.to_csv(TEST_PATH, index = False)
            logger.info(f"Data saved as follow : train : {TRAIN_PATH} and test : {TEST_PATH}")
            logger.info("Data spliting and saving done")
        except Exception as e:
            logger.error(f"Error while data spliting and saving {e}")
            raise CustomException(str(e), sys)
    
    def run(self):
        try:
            logger.info("Data ingestion pipeline started.....")
            df = self.extract_data()

            self.save_data(
                        df = df
                        )
            
            logger.info("End of Data ingestion pipeline...")
        except Exception as e:
            logger.error(f"Error while saving the data {e}")
            raise CustomException(str(e), sys)
        
if __name__ == "__main__":
    data_ingestion_obj = DataIngestion(DB_CONFIG, RAW_DIR)
    data_ingestion_obj.run()






