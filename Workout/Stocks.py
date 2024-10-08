import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# Handling the Figure Window Dimensions
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Inserting CSV
df = pd.read_csv("Beltone_Hackathon\stocks_prices_and_volumes.csv")
# Setting Up Date and Assign fig to the window and setting axis and filling 0 to NAN
fig, ax = plt.subplots()
df.fillna(0, inplace=True)
df.Date = pd.to_datetime(df.Date)
# Conditions
sum_of_volumes = df.iloc[:, 16:31].sum(axis=1)

sum_of_volumexprices=pd.DataFrame()
sum_of_volumexprices=df.iloc[:,1]*df.iloc[:,16]
for i in range(2,15):
    sum_of_volumexprices+=df.iloc[:,i]*df.iloc[:,16+i]

y=sum_of_volumexprices/sum_of_volumes
# Slope Calc
y_clean = np.nan_to_num(y, nan=0.0)
x= np.arange(len(y_clean))
slope, intercept = np.polyfit(x,y_clean,1)
# Plot Data
plt.plot(df.Date,y)
plt.plot(df.Date, slope * x + intercept, color='red', label='Line of best fit')

# Show Window
fig.text(0.01, 0, f'[Slope={slope}], [Min={y.min()},at {df.iloc[y.idxmin(),0]}], [Max={y.max()},at {df.iloc[y.idxmax(),0]}]', ha='left', va='bottom',  fontsize=9)
plt.xlabel("Time")
plt.ylabel("Stock Index")
plt.title("Egyptian Stock Market")
plt.show()