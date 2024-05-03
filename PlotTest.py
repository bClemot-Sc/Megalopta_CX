#### Script for evolutionary algorithm of multiple goal integration
## Author: Bastien Clémot


### ----- Import packages
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import random
import time

### ----- Quick access to some parameters
agent_noise = 10
simulation_border = 200
nb_individuals = 100
nb_generations = 1_000_000
connectivity_anthropy = 0.1
neuron_anthropy = 0.05
num_core = cpu_count()
max_real_time = 3600 * 10


### ----- Define helper functions

## Sinusoid shape function
def sinusoid_shape(direction, ratio):
    x = np.arange(16)
    a = -0.56
    b = 0.81
    c = -4.20
    d = 0.42
    y = a * np.sin(b * (x + c)) + d
    y = list(y * ratio)
    while max(y) != y[direction-1]:
        y.append(y.pop(0))
    y = [i if i >= 0 else 0 for i in y]
    return y

## Assign heading direction to an area
def heading_direction(orientation):
    relative_heading = (-orientation) % 360
    heading_list = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    closest_heading = min(heading_list, key=lambda x: abs(x - relative_heading))
    heading_id = heading_list.index(closest_heading % 360) + 1
    return heading_id

## Linear neuron activation
def linear_activation(activity):
    return np.clip(activity, 0, 1, out=activity)

## Update agent's orientation
def update_orientation(orientation, rotational_speed, noise_deviation):
    random_component = random.gauss(0,noise_deviation)
    new_orientation = orientation + rotational_speed +  random_component
    return new_orientation % 360

## Update agent's position
def update_position(x,y, orientation):
    new_x = x + (math.cos(math.radians(orientation)))
    new_y = y + ( math.sin(math.radians(orientation)))
    return new_x, new_y

## Calculate euclidean distance
def euclidean_distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

## Calculate angle difference between two points
def angle_differences(x, y, goal):
    vector_angle = math.atan2(y, x) * 180 / math.pi
    vector_angle = (vector_angle + 360) % 360
    angle_diff = goal - vector_angle
    angle_diff = (angle_diff + 180) % 360 - 180
    return abs(angle_diff)


### ----- Define evolutionary functions

## Fitness function
def fitness_function(individual_matrix):

    '''Initialisation'''
    # Agent's position list
    agent_list = []
    agent_list.append([0,0,0]) #X, Y, Orientation
    # Neuron activity dataframe
    neuron_df = np.empty((0, individual_matrix.shape[0]))
    neuron_df = np.vstack([neuron_df, np.zeros((1, individual_matrix.shape[0]))])
    # Random goal directions
    ratio1 = random.random()
    ratio2 = 1 - ratio1
    direction1, direction2 = random.sample(range(1, 9), 2)
    best_direction = direction1 if ratio1 > 0.5 else direction2

    '''Simulation loop'''
    i = 0
    while True:
        # Heading direction input to Delta7s
        neuron_df[i,0:16] = sinusoid_shape(heading_direction(agent_list[i][2]),1)
        # Goal direction input to PFLs (two goals)
        neuron_df[i,16:32] = sinusoid_shape(direction1,ratio1)
        neuron_df[i,32:48] = sinusoid_shape(direction2,ratio2)
        # Activity propagation
        neuron_df = np.vstack([neuron_df, linear_activation(np.dot(individual_matrix.T,neuron_df[i,:]))])
        agent_list.append([0,0,0])
        # Heating period
        if i <= 20:
            i += 1
            continue
        # Get rotation from PFLs
        rotation = (np.sum(neuron_df[i+1, -16:-9]) - np.sum(neuron_df[i+1, -8:])) * 10
        # Update agent's orientation
        orientation = update_orientation(agent_list[i][2], rotation, noise_deviation=agent_noise)
        # Update agent's position
        x,y = update_position(agent_list[i][0], agent_list[i][1], orientation)
        # Update agent's dataframe
        agent_list[i+1] = [x,y,orientation]
        
        '''Reward'''
        # check if the agent has reached the end
        if euclidean_distance(0,0,x,y) >= simulation_border:
            distance_reward = simulation_border 
            direction_reward = 180 - angle_differences(x,y, math.radians(45 * (best_direction-1)))
            time_reward = 500 - i
            break
        if i > 500:
            distance_reward = euclidean_distance(0,0,x,y)
            direction_reward = 180 - angle_differences(x,y, 45 * (best_direction-1))
            time_reward = 0
            break
        i += 1

    return distance_reward + direction_reward + time_reward,agent_list, direction1, direction2, best_direction


df = pd.read_excel("Optimization_algorithm\Evolutionary_matrix.xlsx", header=None)
numpy_matrix = df.to_numpy()
score, agent_list, direction1, direction2, best_direction = fitness_function(numpy_matrix)

print(direction1)
print(direction2)
print(best_direction)

# Convert angle to radians
direction1 = np.deg2rad((direction1-1)*45)
direction2 = np.deg2rad((direction2-1)*45)
best_direction = np.deg2rad((best_direction-1)*45)

# Calculate the endpoint of the first vector
x1 = 200 * np.cos(direction1)
y1 = 200 * np.sin(direction1)
x2 = 200 * np.cos(direction2)
y2 = 200 * np.sin(direction2)
x3 = 200 * np.cos(best_direction)
y3 = 200 * np.sin(best_direction)

# Plot the first vector
plt.plot([0, x1], [0, y1], label='Vector 1', color="lightblue")
plt.plot([0, x2], [0, y2], label='Vector 2', color = 'lightblue')
plt.plot([0, x3], [0, y3], label='Vector 3', color = 'red')

x_values = []
y_values = []

x_values = [pos[0] for pos in agent_list]
y_values = [pos[1] for pos in agent_list]

# Plot the agent's movements
plt.plot(x_values, y_values, color='black', label='Agent')

# Adding labels to the axes and a title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the plot
plt.legend()
plt.grid(True)
plt.show()