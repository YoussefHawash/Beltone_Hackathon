import pandas as pd
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,f1_score
from sklearn.preprocessing import RobustScaler  # Changed to RobustScaler

from scipy import stats
from Scripts import DataAnalysis, Splitting
from Pickles import Model, Merging


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
    X_train, X_test, Y_train, Y_test = Splitting.split_data(gold_data)
    first,second=Model.train(X_train,Y_train)
    voting_model= Merging.voting(first,second, X_train, Y_train)



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
