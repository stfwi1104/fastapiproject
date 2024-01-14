import requests
import json
from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

# feature input example for prediction (<50k)
featureinput1 =  { 'age':20,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Masters",
            'education-num':16,
            'marital-status':"Separated",
            'occupation':"Sales",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Male",
            'capital-gain':0,
            'capital-loss':0,
            'hours-per-week':50,
            'native-country':"United-States"
            }


# feature input example for prediction (>50k)
featureinput2 =  { 'age':50,
            'workclass':"Private", 
            'fnlgt':234721,
            'education':"Masters",
            'education-num':16,
            'marital-status':"Separated",
            'occupation':"Sales",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Male",
            'capital-gain':1000,
            'capital-loss':0,
            'hours-per-week':50,
            'native-country':"United-States"
            }


# test statuscode on root
def test_api_locally_get_root():
    response = client.get("/")
    assert response.status_code == 200

# test content on root
def test_api_locally_get_root():
    response = client.get("/")
    assert response.json()[0] == 'Welcome to the Project Application'

# test statuscode on inference
def test_api_inference():
    response = client.post('/prediciton/', data=json.dumps(featureinput1))
    assert response.status_code == 200

# test correct prediction on inference: <50k
def test_api_inference_prediction_0():
    response = client.post('/prediciton/', data=json.dumps(featureinput1))
    assert response.json()["Prediction"] == '<50k' 
    
# test correct prediction on inference: >50k    
def test_api_inference_prediction_1():
    response = client.post('/prediciton/', data=json.dumps(featureinput2))
    assert response.json()["Prediction"] == '>50k' 

