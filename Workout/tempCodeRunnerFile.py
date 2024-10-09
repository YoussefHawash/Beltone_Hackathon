x= np.arange(len(y))
# slope, intercept = np.polyfit( x , y, 1)
# # Plot Data
# plt.scatter(df.Date,y,s=3) # Scatter Data
# #plt.plot(df.Date,y) # Plot Data

# plt.plot(df.Date, slope * x + intercept, color='red', label='Line of best fit')

# # Text and Show window
# fig.text(0.01, 0, f'[Slope={slope}], [Min={y.min()},at {df.iloc[y.idxmin(),0]}], [Max={y.max()},at {df.iloc[y.idxmax(),0]}]', ha='left', va='bottom',  fontsize=9)
# plt.xlabel("X-Label")
# plt.ylabel("Y-Label")
# plt.title("Title")
# plt.show()