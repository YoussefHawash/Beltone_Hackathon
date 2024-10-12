import numpy as np
import pandas as pd
import os 
from scipy import stats

def Oil_Average(df):
    df.loc[df[df.columns[1]] < 0, df.columns[1]] =  df.loc[df[df.columns[1]] < 0, df.columns[2]]
    df['AVG'] =(df.iloc[:, 1] +df.iloc[:, 2])/2
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
    for i in range(len(df['24K'])):
        item = df.loc[i,'Timestamp']
        df.loc[i,'Timestamp'] = item[:10]
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    intraday_gold_last_per_day = df.groupby(df['Timestamp'].dt.date).last().reset_index(drop=True)
    return intraday_gold_last_per_day.iloc[:, 1]
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
    
def remove_outliers_zscore(df, columns, threshold=3):
    for col in columns:
        # Apply Z-score to the non-NaN values, then filter based on the Z-score
        z_scores = stats.zscore(df[col].dropna())
        abs_z_scores = abs(z_scores)
        
        # Create a boolean mask for values within the threshold
        mask = abs_z_scores < threshold
        
        # Apply mask back to the DataFrame
        df = df[df.index.isin(df[col].dropna().index[mask])]
    return df

# Apply Z-score method to remove outliers
def CreateFinal(a):
    df=pd.DataFrame()
    df['Date'] = pd.date_range(start='2020-01-01', periods=len(Oil_Average(a[0])), freq='D')
    df['gold_prices']=Intraday(a[9]) 
    df['pct_change']=pct_calc(df)
    df['Oil_AVG']=Oil_Average(a[0])
    df['EFFR']=EFFR(a[1])
    df['ECIR']=ECIR(a[2]) 
    df['Housing_index']=Housing_index(a[3]) 
    df['Inflation_mom']=Inflation_mom(a[4]) 
    df['Inflation_yoy']=Inflation_yoy(a[5]) 
    df=pd.concat([df, Sotcks(a[6])], axis=1)
    df['Vix']=Vix(a[7]) 
    df['Vxeem']=Vxeem(a[8]) 
    

    return df
