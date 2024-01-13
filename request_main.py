import requests
import json

# feature input example for prediction 
featureinput =  { 'age':20,
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



response1 = requests.get('https://https://fastapiproject-0125e24518e6.herokuapp.com/')
response2 = requests.post('https://https://fastapiproject-0125e24518e6.herokuapp.com/prediciton/', data=json.dumps(featureinput))

print(response1.status_code)
print(response1.json())
print(response2.status_code)
print(response2.json())
