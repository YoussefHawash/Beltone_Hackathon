import os
import argparse
import pandas as pd
import numpy as np
from Scripts import DataAnalysis, Splitting,Model, MergingModels,Train,Test


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
   
    #DEV
    Train.Train(crude_oil_prices,federal_rates,corridor_rates,housing_index,inflation_mom,inflation_yoy,stock_prices,vix_indices,vixeem_indices,gold_prices)
    
    #PROD
    # finalmodel=joblib.load('voting_regressor_model.pkl')
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
