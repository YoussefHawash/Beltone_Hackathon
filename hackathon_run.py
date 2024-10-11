import pandas as pd
import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler  # Changed to RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from scipy import stats
from Scripts import DataAnalysis
from statsmodels.tsa.seasonal import seasonal_decompose

def set_date_index(df, date_column):
    """
    Converts a specified date column to a datetime format, sets it as the index,
    and removes the original date column if it's not named 'Date'.

    Parameters:
    df (pd.DataFrame): The DataFrame that contains the date column.
    date_column (str): The name of the column to convert to datetime and set as index.
    """
    df['Date'] = pd.to_datetime(df[date_column])
    df['Date'] = df['Date'].dt.date
    if date_column != 'Date':
        df.drop(columns=[date_column], inplace=True)
    df.set_index('Date', inplace=True)


def main(input_path, output_path):
    """
    Main function to load datasets, process them, and perform regression prediction on crude oil prices.

    Parameters:
    input_path (str): Path to the directory containing the input CSV files.
    output_path (str): Path to save the output CSV file with predictions.
    """
    # Load datasets


    crude_oil_prices = pd.read_csv(os.path.join(input_path, 'crude_oil_prices.csv'))
    federal_rates = pd.read_csv(os.path.join(input_path, 'effective_federal_funds_rate.csv'))
    corridor_rates = pd.read_csv(os.path.join(input_path, 'egyptian_corridor_interest_rates.csv'))
    housing_index = pd.read_csv(os.path.join(input_path, 'housing_index.csv'))
    inflation_mom = pd.read_csv(os.path.join(input_path, 'inflation_month_on_month.csv'))
    inflation_yoy = pd.read_csv(os.path.join(input_path, 'inflation_year_on_year.csv'))
    news_data = pd.read_csv(os.path.join(input_path, 'news.csv'))
    stock_prices = pd.read_csv(os.path.join(input_path, 'stocks_prices_and_volumes.csv'))
    vix_indices = pd.read_csv(os.path.join(input_path, 'vix_index.csv'))
    vixeem_indices = pd.read_csv(os.path.join(input_path, 'vxeem_index.csv'))
    gold_prices = pd.read_csv(os.path.join(input_path, 'intraday_gold.csv'))

    gold_data=DataAnalysis.CreateFinal([crude_oil_prices,federal_rates,corridor_rates,housing_index,inflation_mom,inflation_yoy,stock_prices,vix_indices,vixeem_indices,gold_prices])

    gold_data.dropna(inplace=True)

# Decompose the time series for each feature
    X = gold_data.drop(['Date','gold_prices'], axis=1)
    for column in X.columns:
        decomposition = seasonal_decompose(X[column], model='additive', period=30, extrapolate_trend='freq')
        gold_data[f'{column}_trend'] = decomposition.trend
        gold_data[f'{column}_seasonal'] = decomposition.seasonal
        gold_data[f'{column}_residual'] = decomposition.resid

    # Drop any NaN values introduced during decomposition
    gold_data.dropna(inplace=True)
 



    # Dropping rows with missing values (caused by lag/moving average features)
    # gold_data.dropna(inplace=True)

    # Outlier detection and removal using Z-score
    # Define a threshold for Z-score
    threshold = 3
    z_scores = np.abs(stats.zscore(gold_data.select_dtypes(include=[np.number])))
    gold_data = gold_data[(z_scores < threshold).all(axis=1)]

    # Prepare features and target variable
    X = gold_data.drop(['Date', 'gold_prices'], axis=1)
    Y = gold_data['gold_prices']

    # Split dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scale features using RobustScaler
    scaler = RobustScaler()  # Use RobustScaler instead of StandardScaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    random_search.fit(X_train_scaled, Y_train)

    # Get the best model from RandomizedSearchCV
    best_model = random_search.best_estimator_

    # Initialize the KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)

    # Fit the KNN model
    knn_model.fit(X_train_scaled, Y_train)

    # Make predictions with both models
    y_pred_xgb = best_model.predict(X_test_scaled)
    y_pred_knn = knn_model.predict(X_test_scaled)

    # Combine predictions using Voting Regressor
    voting_model = VotingRegressor(estimators=[('xgb', best_model), ('knn', knn_model)])
    voting_model.fit(X_train_scaled, Y_train)

    # Make predictions with the voting model
    Y_pred_voting = voting_model.predict(X_test_scaled)

    # Calculate RMSE, MAE, and R^2 Score for the voting model
    rmse_voting = np.sqrt(mean_squared_error(Y_test, Y_pred_voting))
    mae_voting = mean_absolute_error(Y_test, Y_pred_voting)
    r2_voting = r2_score(Y_test, Y_pred_voting)

    # Output performance metrics
    print("Best hyperparameters from RandomizedSearchCV:", random_search.best_params_)
    print("XGBoost RMSE:", np.sqrt(mean_squared_error(Y_test, y_pred_xgb)))
    print("KNN RMSE:", np.sqrt(mean_squared_error(Y_test, y_pred_knn)))
    print("Voting Model RMSE:", rmse_voting)
    print("Voting Model MAE:", mae_voting)
    print("Voting Model R^2 Score:", r2_voting)

    # Calculate actual and predicted directions
    actual_direction = np.sign(Y_test.values)
    predicted_direction_voting = np.sign(Y_pred_voting)  # Use predictions from the Voting Regressor

    # Calculate MDA
    mda = np.mean(actual_direction == predicted_direction_voting)
    print('MDA:', mda)

    # Feature Importance Analysis
    importance = best_model.get_booster().get_score(importance_type='weight')

    # Convert to DataFrame for easier viewing
    importance_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)

    # Display feature importance as numbers
    print("\nFeature Importance from XGBoost:")
    print(importance_df)

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(Y_test.values, color='blue', label='Actual Values')
    plt.plot(Y_pred_voting, color='red', label='Predicted Values (Voting Model)')
    plt.title('Actual vs Predicted Gold Price Change')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Gold Price Change')
    plt.legend()
    plt.show()

    # Plot feature importance as a bar chart
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='green')
    plt.title('Feature Importance (Mean Decrease Accuracy)')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.show()




        # # Create output DataFrame and save to CSV
        # output_df = pd.DataFrame({
        #     'date': features_df.index,
        #     'prediction': y_pred.flatten()
        # })

        # output_df.to_csv(output_path, index=False)
        # print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    # Argument parser to get input and output paths from command line
    parser = argparse.ArgumentParser(description="Process input and output file paths.")
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory.')
    args = parser.parse_args()
    main(args.input_path, args.output_path)
