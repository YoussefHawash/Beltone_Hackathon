import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
from scipy import stats
from scikeras.wrappers import KerasRegressor

def train(x,y):
  
    
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
    lstm_model = KerasRegressor(model=model, epochs=20, batch_size=32, verbose=1)

    # Reshape the input for LSTM (3D array: [samples, timesteps, features])

    # Fit the LSTM model using all data
    lstm_model.fit(x, y)
    return best_model, lstm_model