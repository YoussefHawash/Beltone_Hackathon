import DataAnalysis
import pickle
import os
import sys
import argparse
from sklearn.ensemble import VotingRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,f1_score
from sklearn.preprocessing import RobustScaler  # Changed to RobustScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from scipy import stats
from scikeras.wrappers  import KerasRegressor


from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from scipy import stats

def split_data(All_data):
   
    # Split dataset
    split_index = int(0.8 * len(All_data))
    train_set = All_data[:split_index]  # First 80% of the data
    test_set = All_data[split_index:]

    X_train = train_set.drop([ 'Date','gold_prices','pct_change'], axis=1)
    Y_train = train_set['pct_change']
    X_test = test_set.drop(['Date','gold_prices','pct_change'], axis=1)
    Y_test = test_set['pct_change']

    

    return X_train, X_test, Y_train, Y_test 

def trainhybrid(x,y):
    # Define an optimized parameter grid for RandomizedSearchCV
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'colsample_bylevel': [0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.3],
        'n_estimators': [100, 200, 300]
    }

    # Initialize the XGBoost model
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Perform randomized search for XGBoost
    random_search = RandomizedSearchCV(
        xgboost_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=4,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Fit the XGBoost model
    random_search.fit(x, y)

    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_
    units=100

    model = Sequential()
    model.add(Input(shape=(x.shape[1], 1)))  # Input layer
    model.add(LSTM(units=units, return_sequences=True))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(50))  # Increased neurons
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Wrap the LSTM model using KerasRegressor
    lstm_model = KerasRegressor(model=model, epochs=100, batch_size=32, verbose=1)

    # Reshape the input for LSTM (3D array: [samples, timesteps, features])

    # Fit the LSTM model using all data
    lstm_model.fit(x, y)

    voting_model = VotingRegressor(estimators=[('xgb', best_model), ('lstm', lstm_model)])
    voting_model.fit(x, y)
    with open('Pickles/pkl1.pkl', 'wb') as file:
     pickle.dump(voting_model, file)

def Test(X_test,Y_test):
    with open('Pickles/pkl1.pkl', 'rb') as file:
        voting_model = pickle.load(file)
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

def Develop(toggle):
    crude_oil_prices = pd.read_csv('InputData/Raw_Data/crude_oil_prices.csv')
    federal_rates = pd.read_csv('InputData/Raw_Data/effective_federal_funds_rate.csv')
    corridor_rates = pd.read_csv('InputData/Raw_Data/egyptian_corridor_interest_rates.csv')
    housing_index = pd.read_csv('InputData/Raw_Data/housing_index.csv')
    inflation_mom = pd.read_csv('InputData/Raw_Data/inflation_month_on_month.csv')
    inflation_yoy = pd.read_csv('InputData/Raw_Data/inflation_year_on_year.csv')
    news_data = pd.read_csv('InputData/Raw_Data/news.csv')
    stock_prices = pd.read_csv('InputData/Raw_Data/stocks_prices_and_volumes.csv')
    vix_indices = pd.read_csv('InputData/Raw_Data/vix_index.csv')
    vixeem_indices = pd.read_csv('InputData/Raw_Data/vxeem_index.csv')
    gold_prices = pd.read_csv('InputData/Raw_Data/intraday_gold.csv')
    
    # gold_data=DataAnalysis.CreateFinal([crude_oil_prices,federal_rates,corridor_rates,housing_index,inflation_mom,inflation_yoy,stock_prices,vix_indices,vixeem_indices,gold_prices]) 
    gold_data=pd.read_csv('InputData\Final.csv')
    X_train, X_test, Y_train, Y_test = split_data(gold_data)
    if toggle:
        trainhybrid(X_train,Y_train)
    Test(X_test,Y_test)
library_path = os.path.abspath('./Libraries')
if library_path not in sys.path:
     sys.path.insert(0, library_path)
Develop(1)