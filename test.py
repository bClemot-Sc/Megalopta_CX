import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd

# Sample list of angles in degrees
angles = [-160, -170, -175, 170, 160, 165, 175]

# Convert angles to radians
angles_radians = np.radians(angles)

# Compute circular mean
mean_angle = circmean(angles_radians)

# Compute circular standard deviation
std_angle = circstd(angles_radians)

# Create a circular plot
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

# Plot the angles
ax.scatter(angles_radians, np.ones_like(angles_radians), marker='o')

# Plot the circular mean
ax.plot([mean_angle, mean_angle], [0, 1], color='r', linewidth=2)

# Add a circular standard deviation indicator
ax.fill_between([mean_angle - std_angle, mean_angle + std_angle], 0, 1, color='orange', alpha=0.3)

# Set the direction of 0 degrees to be counterclockwise
ax.set_theta_direction(1)

# Set the zero location of the angles to be at the bottom
ax.set_theta_zero_location('S')

# Remove radial ticks
ax.set_yticks([])

# Show the plot
plt.show()
