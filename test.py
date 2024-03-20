import numpy as np
from scipy.optimize import curve_fit

# Define sinusoidal function
def sinusoid(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

# Fit and extract signal shape parameters
def fit_sinusoid(activity_vector):
    x = np.arange(len(activity_vector))
    param_sinusoid, _ = curve_fit(sinusoid, x, activity_vector, p0=[1, 2*np.pi/len(x), 0, np.mean(activity_vector)])
    return param_sinusoid

# Sample data
def example_fit():
    import matplotlib.pyplot as plt

    # Sample data
    x = np.arange(16)
    y = np.array([1, 1, 0.28, 0, 0, 0, 0, 1, 1, 1, 0.05, 0, 0, 0, 0.4, 1])  # Example activity vector

    # Fit sinusoid
    params = fit_sinusoid(y)

    # Plot data and fitted sinusoid
    plt.scatter(x, y, label='Data')
    plt.plot(x, sinusoid(x, *params), label='Fitted Sinusoid', color='red')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.title('Fitted Sinusoidal Function')
    plt.show()

    # Print fitted parameters
    print("Fitted Sinusoidal Parameters (a, b, c, d):", params)

# Run example fit
example_fit()
