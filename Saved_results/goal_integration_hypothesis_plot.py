import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import circmean, circstd

# Importing data
DATA = pd.read_excel("Saved_results\Multiple_goal_integration.xlsx", sheet_name="Row_data", header=None)
RATIOS = np.array(list(DATA.iloc[0,:]))
RESULTS = np.array(DATA.iloc[1:,:])

# Jitter function
def jitter(array):
    stdev = .01*(max(array)-min(array))
    return array + np.random.randn(len(array)) * stdev

# Computing mean and standard deviation
mean_angle = []
std_angle = []
for y in range(RESULTS.shape[1]):
    mean_angle.append(np.degrees(circmean(np.radians((RESULTS[:,y] % 360 - 180)/90))))
    std_angle.append(np.degrees(circstd(np.radians((RESULTS[:,y] % 360 -180)/90))))

# Graphical representation
plt.figure(figsize=(10, 6))  # Adjusting figure size
plt.errorbar(RATIOS, mean_angle, yerr=std_angle, fmt='-o', color='#4C72B0', ecolor='#DD8452', capsize=5, label='Mean with Std Dev')
for i, ratio in enumerate(RATIOS):
    plt.scatter(np.full_like((RESULTS[:,i] % 360-180)/90, ratio), jitter((RESULTS[:,i] % 360 -180)/90), color='#55A868', alpha=0.5, label='_nolegend_')
plt.xlabel('Ratio', fontsize=12)  # Adjusting axis label font size
plt.ylabel('Mean', fontsize=12)  # Adjusting axis label font size
plt.title('Mean with Standard Deviation and Jittered Raw Data', fontsize=14, fontweight='bold')  # Adjusting title font size and style
plt.grid(True, linestyle='--', alpha=0.7)  # Adding grid with dashed lines
plt.legend()  # Adding legend
plt.tight_layout()  # Adjusting layout to prevent clipping of labels
plt.show()
