# Importing required libraries
import pandas as pd
import yaml
import pickle
import os
import warnings

# Filtering out warnings
warnings.filterwarnings("ignore")

# Reading the yaml file
params = yaml.safe_load(open("params.yaml"))['test']

# Loading the test data
df = pd.read_csv(params['input'])

# Creating a function to make predictions
def predict(df):

    # Loading the model
    model = pickle.load(open(params['model'], 'rb'))

    # Loading the sclaing transformer
    scaler = pickle.load(open(params['transformers'], 'rb'))

    # Scaling the test data
    scaled_data = scaler.transform(df)

    # Making predictions on the scaled test data
    predictions = model.predict(scaled_data)

    # Adding the predictions to the test data
    df['Predictions'] = predictions

    # Creating a directory to save the predictions
    os.makedirs("data/prediction", exist_ok=True)

    # Saving the predictions in the desired directory
    df.to_csv(params['output'], index=False)

    # Displaying the success statement
    print("The predictions were successfully saved!")

# Defining the code to be executed
if __name__ == '__main__':

    # Calling the predict function with the test data
    predict(df)