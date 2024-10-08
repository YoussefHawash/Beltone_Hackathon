import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# Handling the Figure Window Dimensions
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Inserting CSV
df = pd.read_csv("Workout\Data\crude_oil_prices.csv")
# Setting Up Date and Assign fig to the window and setting axis
df.Date = pd.to_datetime(df.Date)
fig, ax = plt.subplots()
# Conditions
row=df[df.WTI<0].index[0]
df= df.drop(row)
filtered_Europe=df[df.Europe>0]
filtered_WTI=df[df.WTI>0]
y= (filtered_Europe.WTI+filtered_Europe.Europe)/2
# Calc Slope
x= np.arange(len(y))
slope, intercept = np.polyfit( x , y, 1)
# Plot Data
plt.scatter(df.Date,y,s=3)
plt.plot(df.Date, slope * x + intercept, color='red', label='Line of best fit')

# Text and Show window
fig.text(0.01, 0, f'Best Fit Line Slope={slope} , Min={y.min()} , Max={y.max()}', ha='left', va='bottom',  fontsize=9)
plt.xlabel("Time")
plt.ylabel("Dollars per Barrel")
plt.title("Cruide Oil Price {AVG}")
plt.show()