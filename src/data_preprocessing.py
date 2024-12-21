# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import os
import yaml
import pickle

import warnings

# Filtering out warnings
warnings.filterwarnings("ignore")

# Loading the parameters from the yaml file
params = yaml.safe_load(open("params.yaml"))['preprocess']

# Creating a function to preprocess data
def preprocess(input_path, output_path):

    # Loading the raw data
    df = pd.read_csv(params['input'])

    # Dropping NA values
    df = df.dropna()

    # Dropping duplicates
    df = df.drop_duplicates()

    # Creating a scaling object
    scaler = MinMaxScaler(feature_range=(0,1))

    # Splitting the dataframe into data and target values
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scaling the data
    scaled_X = scaler.fit_transform(X)

    # Craeting a scaled dataframe
    scaled_df = pd.DataFrame(scaled_X, columns=X.columns)
    scaled_df["Outcome"] = y

    # Creating a directory to save the preprocessed data
    os.makedirs("../data/preprocessed", exist_ok=True)

    # Creating a directory to store the scaling transformer
    os.makedirs("transformers", exist_ok=True)

    # Saving the scaling transformer in the desired directory
    pickle.dump(scaler, open(params['transformers'], 'wb'))

    # Saving the preprocessed data in the desired directory
    scaled_df.to_csv(output_path, index=False)

    # Displaying the success statement
    print("The data was successfully preprocessed!")

# Defining the code to be executed
if __name__ == '__main__':

    # Calling the preprocess function with the input path
    preprocess(params['input'], params['output'])