
import pandas as pd
import pickle
import os
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics


# load model, encoder and binarizer
model = pickle.load(open("model/model.pkl",'rb'))   
encoder = pickle.load(open("model/encoder.pkl",'rb')) 
lb = pickle.load(open("model/lb.pkl",'rb'))  

# Load census data into Dataframe
data = pd.read_csv("data/census.csv")
for column in data.columns:
    data = data.rename(columns={column : column.strip()})

# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Calculate Model-Metrics Slicing
# Loop through all categorical features
for feature in cat_features:
    # Loop througg all distinct values of cat feature
    for cls in data[feature].unique():
        data_sclice = data[data[feature] == cls]
        X_slice, y_sclice, encoder, lb = process_data(
        data_sclice, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
        )        
        preds_sclice = model.predict(X_slice)
        precision, recall, fbeta = compute_model_metrics(y_sclice,preds_sclice)
        print(f"feature: {feature} value: {cls} precision: {precision}")
        # write to txt
        with open('slice_output.txt', 'a') as f:
            f.write(f"feature: {feature} value: {cls} precision: {precision}\n") 
      