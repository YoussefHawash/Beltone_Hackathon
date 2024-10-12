import pickle
import time
from sklearn.metrics import mean_squared_error, f1_score
import numpy as np
import pandas as pd


def Model(X, test=None):
    """
    Load a pre-trained linear regression model from a pickle file and use it to make predictions on input data.

    Parameters:
    X (pd.DataFrame or np.ndarray): The input features for which predictions are to be made.
    test (pd.DataFrame or np.ndarray, optional): The true values to test model performance (if provided).

    Returns:
    np.ndarray: The predicted values based on the input features.
    """
    # Load the pre-trained linear regression model
    with open('Pickles/pkl1.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Predict based on the input features
    y_pred = loaded_model.predict(X)

    # If test data is provided, perform model validation
    if test is not None:
        validation(loaded_model, X, test)

    return y_pred


def validation(model, X_test, y_test):
    """
    Validate the performance of the regression model using test data and print evaluation metrics.

    Metrics include:
    - Root Mean Squared Error (RMSE)
    - Mean Directional Accuracy (MDA)
    - Bucketized F1 Score (using quartiles)
    - Inference Time

    Parameters:
    model (object): The regression model used for predictions.
    X_test (pd.DataFrame or np.ndarray): The input test features.
    y_test (pd.DataFrame or np.ndarray): The true values for the test set.
    """
    # Measure inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Calculate Mean Directional Accuracy (MDA)
    mda = np.mean(np.sign(np.diff(y_test)) == np.sign(np.diff(y_pred)))

    # Bucketize the true and predicted values into quartiles and calculate F1 score
    y_test_buckets = pd.qcut(y_test, 4, labels=False)
    y_pred_buckets = pd.qcut(y_pred, 4, labels=False)
    f1 = f1_score(y_test_buckets, y_pred_buckets, average='weighted')

    # Print the evaluation metrics
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Directional Accuracy: {mda}')
    print(f'Bucketized F1 Score: {f1}')
    print(f'Inference Time (seconds): {inference_time}')
