import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import GradientBoosting 
# Handling the Figure Window Dimensions
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Setting Up Date and Assign fig to the window and setting axis
fig, ax = plt.subplots()

# Plot Data

plt.plot(pd.DataFrame(GradientBoosting.X_test),pd.DataFrame(GradientBoosting.y_pred),color='red')

plt.plot(pd.DataFrame(GradientBoosting.X_test),pd.DataFrame(GradientBoosting.y_test),color='green')


# Text and Show window
plt.xlabel("Time")
plt.ylabel("Prec Change")
plt.title("Test")
plt.show()