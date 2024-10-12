import joblib
from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,f1_score
def Test(X_test,Y_test):
    voting_model = joblib.load('Pickles/voting_regressor_model.pkl')
    Y_pred_voting = voting_model.predict(X_test)

    # Calculate RMSE, MAE, and R^2 Score for the voting model
    rmse_voting = np.sqrt(mean_squared_error(Y_test, Y_pred_voting))
    mae_voting = mean_absolute_error(Y_test, Y_pred_voting)
    r2_voting = r2_score(Y_test, Y_pred_voting)
    # Output performance metrics
    print("RMSE:", rmse_voting)
    print("MAE:", mae_voting)
    print("R^2", r2_voting)
    # Calculate actual and predicted directions
    actual_direction = np.sign(Y_test.values)
    predicted_direction_voting = np.sign(Y_pred_voting)  # Use predictions from the Voting Regressor

    # Calculate MDA
    mda = np.mean(actual_direction == predicted_direction_voting)
    print('MDA:', mda)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test.values, color='blue', label='Actual Values')
    plt.plot(Y_pred_voting, color='red', label='Predicted Values (Voting Model)')
    plt.title('Actual vs Predicted Gold Price Change')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Gold Price Change')
    plt.legend()
    plt.show()
