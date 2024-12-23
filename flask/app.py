from flask import Flask, render_template, request, jsonify
from flask_httpauth import HTTPBasicAuth
import pickle
import numpy as np
import json
import logging

# Load model and load Auth keys
HOME_PATH = '.' #/home/ec2-user/Documents/GitHub/my-web-server/flask'
PATH_MODEL = f'{HOME_PATH}/models/random_forest_model.pkl'
PATH_AUTH = f'{HOME_PATH}/config/auth_flask.json'
try:
    with open(PATH_MODEL, 'rb') as file:
        model_obj = pickle.load(file)
except FileNotFoundError:
    logging.error(f"Model file not found at {PATH_MODEL}")
    raise
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise

try:
    with open(PATH_AUTH, 'r') as file:
        users = json.load(file)
except FileNotFoundError:
    logging.error(f"Auth file not found at {PATH_AUTH}")
    raise
except Exception as e:
    logging.error(f"Error loading auth file: {str(e)}")
    raise

app = Flask('Score transactions')
auth = HTTPBasicAuth()

# Setup logging
logging.basicConfig(filename='logs/flask.log', level=logging.DEBUG, format='%(asctime)s %(message)s')

@app.before_request
def log_request_info():
    logging.debug(f"Headers: {request.headers}")
    logging.debug(f"Body: {request.get_data()}")

# Setup password authentication
@auth.verify_password
def verify_password(username, password):
    logging.debug(f"Auth attempt with username={username} and password={password}")
    if username == users['prod']['user'] and password == users['prod']['password']:
        logging.debug("Authentication successful")
        return username
    logging.warning(f"Failed auth attempt with username={username}")

# Setup predict function
@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    try:
        data_input = request.get_json()
        if not data_input:
            raise ValueError("No input data provided")
        data_np = np.array(list(data_input.values())).reshape(1, -1)
        y_proba = model_obj.predict_proba(data_np)
        predicted_class = np.argmax(y_proba[0])
        predicted_probability = y_proba[0][predicted_class]
        result = {
            'model_name': 'random_forest_model',
            'score': predicted_probability
        }
        logging.debug(f"Prediction result: {result}")
        return jsonify(result), 200
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return render_template(f"index.html")

#cd my-web-server/flask && python3.9 app.py
if __name__ == "__main__": 
    app.run(host="0.0.0.0", port=8502, debug=True)
