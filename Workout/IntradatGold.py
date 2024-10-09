import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
# Handling the Figure Window Dimensions
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Inserting CSV
df = pd.read_csv("Workout\Data\intraday_gold.csv")
# Setting Up Date and Assign fig to the window and setting axis
def parse_mixed_formats(ts):
    if 'T' in ts and '+' in ts:
        return pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S%z++00:00', errors='coerce')
    elif '+' in ts:
        return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S%z++00:00', errors='coerce')
    else:
        return pd.to_datetime(ts, errors='coerce')

# Apply the custom function
df['Timestamp'] = df['Timestamp'].apply(parse_mixed_formats)
fig, ax = plt.subplots()
y=df.twentyfourK # Define Your y
# Calc Slope
x= np.arange(len(y))
slope, intercept = np.polyfit( x , y, 1)
# Plot Data
plt.scatter(df.Timestamp,y,s=5) # Scatter Data
#plt.plot(df.Date,y) # Plot Data

plt.plot(df.Timestamp, slope * x + intercept, color='red', label='Line of best fit')

# Text and Show window
fig.text(0.01, 0, f'[Slope={slope}], [Min={y.min()},at {df.iloc[y.idxmin(),0]}], [Max={y.max()},at {df.iloc[y.idxmax(),0]}]', ha='left', va='bottom',  fontsize=9)
plt.xlabel("Time")
plt.ylabel("24K")
plt.title("IntraDay Gold")
plt.show()