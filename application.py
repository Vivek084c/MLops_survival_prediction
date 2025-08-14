import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

from prometheus_client import start_http_server, Counter, Gauge

from config.paths_config import MODEL_PATH
from alibi_detect.cd import KSDrift
from src.feature_store import RedisFeatureStore
from sklearn.preprocessing import StandardScaler
from src.logger import get_logger

logger = get_logger(__name__)

app = Flask(__name__, template_folder="templates")

prediction_count = Counter('prediction_count', "Number of prediction counte")
drift_count = Counter('drift_counter',"Number of times the drift is detected")

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

FEATURE_NAMES = ['Age', 'Fare', 'Pclass', 'Sex', 'Embarked', 'Familysize', 'Isalone',
    'HasCabin', 'Title', 'Pclass_Fare', 'Age_Fare']

feature_store = RedisFeatureStore()
scaler = StandardScaler()

def fit_scaler_on_referance_data():
    ids = feature_store.get_all_entity_ids()
    all_features = feature_store.get_batch_features(ids)

    data_df = pd.DataFrame.from_dict(all_features, orient='index')[FEATURE_NAMES]

    scaler.fit(data_df)
    return scaler.transform(data_df)  # <-- Use data_df, not all_features

historical_data = fit_scaler_on_referance_data()
ksd = KSDrift(x_ref=historical_data)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        data = request.form
        Age = float(data["Age"])
        Fare = float(data["Fare"])
        Pclass = int(data["Pclass"])
        Sex = int(data["Sex"])
        Embarked = int(data["Embarked"])
        Familysize = int(data["Familysize"])
        Isalone = int(data["Isalone"])
        HasCabin = int(data["HasCabin"])
        Title = int(data["Title"])
        Pclass_Fare = float(data["Pclass_Fare"])
        Age_Fare = float(data["Age_Fare"])

        features = pd.DataFrame([[Age, Fare, Pclass, Sex, Embarked, Familysize,Isalone , HasCabin, Title, Pclass_Fare, Age_Fare]])

        ### data drift predictions
        feature_salced = scaler.transform(features)        


        drift = ksd.predict(feature_salced)
        print("Drift Response : ", drift)

        drift_response = drift.get('data', {})
        is_drift = drift_response.get('is_drift', None)
        if is_drift is not None and is_drift==1:
            drift_count.inc()
            print("drift detected...")
            logger.info(f"Data drift is found")


        prediction = model.predict(features)[0]
        prediction_count.inc()

        result = "Survived" if predict==1 else "Did not Survived"
        return render_template('index.html', prediction_text = f"The predictions is : {result}")
    except Exception as e:
        return jsonify({'error':str(e)})
    
@app.route('/mertics')
def metrics():
    from prometheus_client import generate_latest
    from flask import Response

    return Response(generate_latest(), content_type='text/plain')

if __name__ == "__main__":
    # runing the promethus server
    start_http_server(8000)
    app.run(debug=True, host='0.0.0.0', port=5010)



