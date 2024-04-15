import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DATA = pd.read_excel("Saved_results\Multiple_goal_integration.xlsx", sheet_name="Row_data", header=None)
RATIOS = np.array(list(DATA.iloc[0,:]))
RESULTS = np.array(DATA.iloc[1:,:].T)

means = np.mean(RESULTS % 360, axis=1)
print(means)