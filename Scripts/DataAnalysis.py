import numpy as np
import pandas as pd
import os 
from scipy import stats

def Oil_Average(df):
    df.loc[df[df.columns[1]] < 0, df.columns[1]] =  df.loc[df[df.columns[1]] < 0, df.columns[2]]
    df['AVG'] =(df.iloc[:, 1] +df.iloc[:, 2])/2
    print(df['AVG'] )
    return df['AVG'] 
def EFFR(df):
    df.EFFR= pd.to_numeric(df.EFFR, errors='coerce').fillna(0).astype(float)
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
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)

    # Drop rows with invalid timestamps (if any)
    df = df.dropna(subset=['Timestamp'])

    # Extract the date from the 'Timestamp' column
    df['Date'] = df['Timestamp'].dt.date

    # Group by the date and get the last (closing) price for each day
    closing_prices_df = df.groupby('Date').last().reset_index()

    # Select only the 'Date' and '24K' (closing price) columns
    closing_prices_df = closing_prices_df[['Date', '24K']]
    return closing_prices_df.iloc[:, 1]
def Sotcks(df_stocks):
    new_column_names = []
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
    dfc['gold_prices_shifted'] = dfc['gold_prices'].shift(-1)
    dfc['pct_change'] = (dfc['gold_prices_shifted'] - dfc['gold_prices']) / dfc['gold_prices'] * 100
    dfc.dropna(subset=['pct_change'], inplace=True)
    return dfc['pct_change']
    


# Apply Z-score method to remove outliers
def CreateFinal(a):
    df=pd.DataFrame()
    df['gold_prices']=Intraday(a[9]) 
    df['pct_change']=pct_calc(df)
    df['Oil_AVG']=Oil_Average(a[0])
    print (df['Oil_AVG'])
    df['EFFR']=EFFR(a[1])
    df['ECIR']=ECIR(a[2]) 
    df['Housing_index']=Housing_index(a[3]) 
    df['Inflation_mom']=Inflation_mom(a[4]) 
    df['Inflation_yoy']=Inflation_yoy(a[5]) 
    # df=pd.concat([df, Sotcks(a[6])], axis=1)
    df['Vix']=Vix(a[7]) 
    df['Vxeem']=Vxeem(a[8]) 
    print(df)
    return df
