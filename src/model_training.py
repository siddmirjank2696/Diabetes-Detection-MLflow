# Importing the required libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import mlflow
from mlflow.models import infer_signature

import os
import yaml
import pickle

import warnings

# Filtering out warnings
warnings.filterwarnings("ignore")

# Setting mlflow configuration with dagshub
os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/siddmirjank2696/Diabetes-Detection-MLflow.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="siddmirjank2696"
os.environ["MLFLOW_TRACKING_PASSWORD"]="ed2891a2b1abb4d15f5d9a6a45f8bb11829b6bed"

# Accessing the contents of the yaml file
params = yaml.safe_load(open("params.yaml"))['train']

# Loading the data
df = pd.read_csv(params['input'])

# Splitting the data and labels
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Splitting the data and labels into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=params['random_state'])

# Creating a function to perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, X_test):

    # Creating a parameters grid
    param_grid = {
        'n_estimators' : [100, 200, 300],
        'criterion' : ['gini', 'entropy', 'log_loss'],
        'max_features' : ['sqrt', 'log2', None]
    }

    # Creating a model object
    rf = RandomForestClassifier(random_state=params['random_state'])

    # Creating a GridSearchCV object
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1, cv=3)

    # Fitting the GridSearchCV object to the training data
    grid_search.fit(X_train, y_train)

    # Returning the grid search object
    return grid_search

# Creating a function to train a Machine Learning Model
def train(X_train, y_train, X_test):

    # Setting mlflow tracking uri
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

    # Setting mlflow experiment name
    mlflow.set_experiment(params['experiment_name'])

    # Defining the model signature
    signature = infer_signature(X_train, y_train)

    # Starting mlflow experiment
    with mlflow.start_run():

        # Performing hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, X_test)

        # Retrieving the best model
        best_model = grid_search.best_estimator_

        # Retrieving the best parameters
        best_params = grid_search.best_params_

        # Logging the best parameters
        mlflow.log_params(best_params)

        # Making predictions on the unseen test data
        y_pred = best_model.predict(X_test)

        # Calculating the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        # Logging the accuracy
        mlflow.log_metric("accuracy", accuracy)

        # Logging the model
        model_info = mlflow.sklearn.log_model(
            sk_model = best_model,
            artifact_path = "diabetes_model",
            signature = signature
        )

    # Creating a directory to store the best model
    os.makedirs("models", exist_ok=True)

    # Saving the best model in the desired directory
    pickle.dump(best_model, open(params['output'], 'wb'))

    # Displaying the success statement
    print("\nThe model was successfully saved!")

# Definining the code to be executed
if __name__ == '__main__':

    # Calling the train function with the training data
    train(X_train, y_train, X_test)