import pandas as pd
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE
import sys

from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import TRAIN_PATH, TEST_PATH

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self, train_data_path, test_data_path, feature_store : RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data = None
        self.test_data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.X_resampled = None
        self.y_resampled = None

        self.feature_store = feature_store

        logger.info("Data Processing Initilised")
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Read the data Successfully")
        except Exception as e:
            logger.error(f"Error in reading the data {e}")
            raise CustomException(str(e), sys)
        
    def preprocess_data(self):
        try:
            self.data["Age"] = self.data["Age"].fillna(self.data["Age"].median())
            self.data["Embarked"] = self.data["Embarked"].fillna(self.data["Embarked"].mode()[0])
            self.data["Fare"] = self.data["Fare"].fillna(self.data["Fare"].median())

            self.data['Sex'] = self.data["Sex"].map({
                'male' : 0,
                'female' : 1
            })
            self.data["Embarked"] = self.data["Embarked"].astype('category').cat.codes

            self.data['Familysize'] = self.data['SibSp'] + self.data["Parch"] + 1
            self.data["Isalone"] = (self.data['Familysize']==1).astype(int) # if familiy ==1 then he is alone
            self.data["HasCabin"] = self.data["Cabin"].notnull().astype(int) # if cabin number is assgined then it is 1 else 0
            self.data["Title"] = self.data["Name"].str.extract(' ([A-Za-z]+)\.', expand=False).map( 
                {
                    'Mr':0, 
                    'Miss':1,
                    'Mrs':2,
                    'Master':3,
                    'Rare':4
                }
            ).fillna(4) # 
            self.data['Pclass_Fare'] = self.data['Pclass'] * self.data['Fare']
            self.data['Age_Fare'] = self.data['Age'] * self.data['Fare']

            logger.info(f"Data Preprocessing Done")
        except Exception as e:
            logger.error(f"Error in Data Preprocessing {e}")
            raise CustomException(str(e), sys)
    
    def handle_imbalance_data(self):
        try:
            X = self.data[['Pclass','Sex', 'Age', 'Fare', 'Embarked', 'Familysize', 'Isalone',  'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']]
            y = self.data['Survived']

            smote = SMOTE()
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)

            logger.info(f"Handeled imbalance data Successfully")

        except Exception as e:
            logger.error(f"Error in Handeled imbalance data Successfully {e}")
            raise CustomException(str(e), sys)
    
    def store_features_in_redis(self):
        try:
            batch_data = {}
            for idx, row in self.data.iterrows():
                entity_id = row["PassengerId"]
                features = {
                    "Age": row['Age'],
                    "Fare": row["Fare"],
                    "Pclass": row["Pclass"],
                    "Sex": row["Sex"],
                    "Embarked": row["Embarked"],
                    "Familysize": row["Familysize"],
                    "HasCabin": row["HasCabin"],
                    "Title": row["Title"],
                    "Pclass_Fare": row["Pclass_Fare"],
                    "Age_Fare": row["Age_Fare"],
                    "Survived": row["Survived"],  # <-- Add this line
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been feature into feature Store..")
        except Exception as e:
            logger.error(f"Error in Data feature storing {e}")
    def retirve_features_from_redis_store(self, entity_id):
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        return None

    def run(self):
        try:
            logger.info("Strating data processing pipeline")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalance_data()
            self.store_features_in_redis()

            logger.info(f"End of data processing pipeline")
        except Exception as e:
            logger.info(f"Errror while data processing pipeline {e}")
            raise CustomException(str(e), sys)
if __name__ == "__main__":
    feature_store = RedisFeatureStore(

    )
    data_processor = DataProcessing(
                        train_data_path= TRAIN_PATH ,
                        test_data_path= TEST_PATH,
                        feature_store=feature_store )

    data_processor.run()

    print(data_processor.retirve_features_from_redis_store(entity_id=332))