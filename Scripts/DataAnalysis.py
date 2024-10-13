import numpy as np
import pandas as pd
import os 
from datetime import datetime
import pytz
from statsmodels.tsa.seasonal import seasonal_decompose

def Oil_Average(df):
    df.loc[df[df.columns[1]] < 0, df.columns[1]] =  df.loc[df[df.columns[1]] < 0, df.columns[2]]
    df['AVG'] =(df.iloc[:, 1] +df.iloc[:, 2])/2
    return df['AVG'] 
def EFFR(df):
    df.iloc[0,1] = 0
    for i in range(1,len(df)):
        if df.iloc[i,1] == '.':
            df.iloc[i,1] = df.iloc[i-1,1]
    df.EFFR= pd.to_numeric(df.EFFR, errors='coerce').astype(float)
    return df.EFFR

def ECIR(df):
    return df.iloc[:, 2]

def Housing_index(df):
    return df.iloc[:, 1]

def Inflation_mom(df):
    return df.iloc[:, 1]
def Inflation_yoy(df):
    return df.iloc[:, 1]
def Intraday(df):
    # Try to convert the 'Timestamp' column to datetime, handling any format issues
    date_formats = [
    '%Y-%m-%d %H:%M:%S',    # 2023-03-15 12:30:45
    '%Y-%m-%d %H:%M:%S.%f', # 2023-03-15 12:30:45.123456
    '%Y-%m-%dT%H:%M:%S%z',  # 2023-03-15T12:30:45+0000 (ISO with timezone)
    '%m/%d/%Y %H:%M:%S',    # 03/15/2023 12:30:45
    '%d-%m-%Y %H:%M:%S',    # 15-03-2023 12:30:45
    '%Y-%m-%d %z', 
    '%Y-%m-%d',             # 2023-03-15 (date only)            # 2023-03-15 (date only)
    '%d-%m-%Y',             # 15-03-2023 (date only)
    '%m/%d/%Y',
    '%m/%d/%Y %z',             # 03/15/2023 (date only)             # 03/15/2023 (date only)
    '%Y.%m.%d %H:%M:%S',    # 2023.03.15 12:30:45
    '%Y/%m/%d %H:%M:%S',    # 2023/03/15 12:30:45
    '%Y-%m-%dT%H:%M:%S.%f%z', # 2023-03-15T12:30:45.123456+0000 (ISO with fractional seconds and timezone)
    ]

    # Custom function to attempt parsing with multiple date formats
    def try_parsing_date(text):
        for fmt in date_formats:
            try:
                # Attempt parsing the date with each format
                dt = datetime.strptime(text, fmt)
                return dt
            except ValueError:
                continue
        return pd.NaT  # If none of the formats match, return NaT (Not a Time)

    # Load the data

    # Apply the custom date parsing function to the 'Timestamp' column
    df['Timestamp'] = df['Timestamp'].apply(lambda x: try_parsing_date(str(x)))

    # Drop rows with invalid 'Timestamp' values (NaT values)
    df = df.dropna(subset=['Timestamp'])

    # Convert timezone-aware datetimes to UTC (or any other desired timezone)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)

    # Extract the date from the 'Timestamp' column
    df['Date'] = df['Timestamp'].dt.date

    # Group by the date and get the last (closing) price for each day
    closing_prices_df = df.groupby('Date').last().reset_index()

    # Select only the 'Date' and '24K' (closing price) columns
    closing_prices_df = closing_prices_df[['Date', '24K']]
    return closing_prices_df.iloc[:, 1]
def Sotcks(df_stocks):
    new_column_names = []
    df_stocks = df_stocks.fillna(0)
    for i in range(15):
        # Construct price and volume column names based on their index
        price_col = df_stocks.columns[i+1]  # Price columns start from index 1 (skipping 'Date')
        volume_col = df_stocks.columns[i+16]  # Volume columns start from index 16
        
        # Generate new column name by modifying the stock name from price column
        new_column_name = f"{price_col.split('_close_price')[0]}_value"
        new_column_names.append(new_column_name)
        
        # Multiply price and volume, and store the result in the new column
        df_stocks[new_column_name] = df_stocks[price_col] * df_stocks[volume_col]
    return df_stocks[new_column_names]
def Vix(df):
    vix_index=(df.iloc[:, 1]+df.iloc[:, 2]+df.iloc[:, 3]+df.iloc[:, 4])/4
    return vix_index
def Vxeem(df):
    vxeem_index=(df.iloc[:, 1]+df.iloc[:, 2]+df.iloc[:, 3]+df.iloc[:, 4])/4
    return vxeem_index
def pct_calc(df):
    dfc=df.copy()
    dfc['gold_prices_shifted'] = dfc['gold_prices'].shift(1)
    dfc['pct_change'] = (dfc['gold_prices_shifted'] - dfc['gold_prices']) / dfc['gold_prices'] * 100
    dfc.dropna(subset=['pct_change'], inplace=True)
    return dfc['pct_change']
    


# Apply Z-score method to remove outliers
def CreateFinal(a):
    df=pd.DataFrame()
    df['Date']=a[0]['Date']
    df['gold_prices']=Intraday(a[9]) 
    df['pct_change']=pct_calc(df)
    df['Oil_AVG']=Oil_Average(a[0])
    df['EFFR']=EFFR(a[1])
    df['ECIR']=ECIR(a[2]) 
    df['Housing_index']=Housing_index(a[3]) 
    df['Inflation_mom']=Inflation_mom(a[4]) 
    df['Inflation_yoy']=Inflation_yoy(a[5]) 
    for i in range(1,15):
        df[f'Stocknum{i}']=Sotcks(a[6]).iloc[:,i]
    
    df['Vix']=Vix(a[7]) 
    df['Vxeem']=Vxeem(a[8]) 
    exclude_columns = ['Date','pct_change', 'gold_prices']
    df.fillna(0, inplace=True)
    for column in df.drop(columns=exclude_columns).columns:
        decomposition = seasonal_decompose(df[column], model='additive', period=30, extrapolate_trend='freq')
        # Reconstruct the final observed values
        df[f'{column}_reconstructed_observed'] = decomposition.trend + decomposition.seasonal + decomposition.resid
    return df
