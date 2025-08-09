import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os


from src.logger import get_logger
from src.custom_exception import CustomException
from src.feature_store import RedisFeatureStore



logger = get_logger(__name__)

class ModelTraining:
    def __init__(
                self,
                feature_store: RedisFeatureStore,
                model_save_path = "artifacts/models/"
                ):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None
    
        os.makedirs(self.model_save_path, exist_ok=True)

        logger.info("Model Training Initialized...")

    def load_data_from_redis(self, entity_ids):
        try:
            logger.info("Extracting data from redids")
            data = []
            for entity_id in entity_ids:
                features = self.feature_store.get_features(entity_id)
                if features:
                    data.append(features)
                else:
                    logger.warning("Feature not found")
            return data
        except Exception as e:
            logger.error(f"Error while Extracting data from redis {e}")
            raise CustomException(str(e), sys)

    def prepare_data(self):
        try:
            entity_id = self.feature_store.get_all_entity_ids()
            train_entity_id, test_entity_id = train_test_split(entity_id, test_size=0.2, random_state=42)

            train_data = self.load_data_from_redis(train_entity_id)
            test_data = self.load_data_from_redis(test_entity_id)

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)
            logger.info(train_df)

            x_train = train_df.drop("Survived", axis=1)
            x_test = test_df.drop("Survived", axis=1)
            y_train = train_df["Survived"]
            y_test = test_df["Survived"]

            logger.info(f"Compleated the Model traininig")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error while preparing the data {e}")
            raise CustomException(str(e), sys)
        
    def hyperparameter_tuning(self, x_train, y_train):
        try:
            param_distribution = {
                    'n_estimators' : [100, 200, 300],
                    'max_depth' : [10, 20, 30],
                    'min_samples_split' : [2,5],
                    'min_samples_leaf' : [1,2]
                    }
            
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                                    rf, 
                                    param_distributions = param_distribution,
                                    n_iter=10,
                                    cv = 3,
                                    scoring='accuracy', 
                                    random_state=42)
            random_search.fit(x_train, y_train)
            logger.info(f"Best params are : {random_search.best_params_}")
            return random_search.best_estimator_

        except Exception as e:
            logger.error(f"Error while hyper parameter tunning {e}")
            raise CustomException(str(e), sys)
    
    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        try:
            best_rf = self.hyperparameter_tuning(x_train, y_train)
            
            ypred = best_rf.predict(x_test)

            score = accuracy_score(y_test, ypred) 
            logger.info(f"Accuracy is {score}")
            self.save_model(best_rf)
            return score
        except Exception as e:
            logger.error(f"Error while training the model {e}")
            raise CustomException(str(e), sys)
        
    def save_model(self, model):
        try:
            model_filename = f"{self.model_save_path}random_forest_model.pkl"
            with open(model_filename, 'wb') as model_file:
                pickle.dump(model, model_file)
            logger.info(f"Model is saved successfully {model_filename}")
            
        except Exception as e:
            logger.error(f"Error in saving the model {e}")
            raise CustomException(str(e), sys)
        
    def run(self):
        try:
            logger.info(f"Strating model traininig pipeline")
            x_train, x_test, y_train, y_test = self.prepare_data()
            accuracy = self.train_and_evaluate(x_train, y_train, x_test, y_test)
            logger.info("End of model traninng pipeline")
        except CustomException as e:
            logger.error(f"Error while model traininig pipeline {e}")
            raise CustomException(str(e), sys)


if __name__=="__main__":
    feature_store = RedisFeatureStore()
    model_traininig = ModelTraining(feature_store)

    model_traininig.run()