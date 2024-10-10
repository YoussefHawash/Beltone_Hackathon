import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# Handling the Figure Window Dimensions
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Inserting CSV
df = pd.read_csv("Workout\Data\inflation_yoy_headline.csv")
# Setting Up Date and Assign fig to the window and setting axis
df.Date = pd.to_datetime(df.Date)
fig, ax = plt.subplots()
y=df.Headline
# Calc Slope
x= np.arange(len(y))
slope, intercept = np.polyfit( x , y, 1)
# Plot Data
plt.scatter(df.Date,y,s=3)
# plt.plot(df.Date,y)

plt.plot(df.Date, slope * x + intercept, color='red', label='Line of best fit')

# Text and Show window
fig.text(0.01, 0, f'[Slope={slope}], [Min={y.min()},at {df.iloc[y.idxmin(),0]}], [Max={y.max()},at {df.iloc[y.idxmax(),0]}]', ha='left', va='bottom',  fontsize=9)
plt.xlabel("Time")
plt.ylabel("y/y")
plt.title("Inflation Year on Year {HeadLine}")
plt.show()