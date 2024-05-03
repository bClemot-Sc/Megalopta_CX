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
from tqdm import tqdm


### ----- Quick access to some parameters
agent_noise = 5
simulation_border = 200
small_ratio = 0.25
high_ratio = 0.75
small_direction, high_direction = random.sample(range(1, 9), 2)
nb_individuals = 4
nb_generations = 20
connectivity_anthropy = 0.2
neuron_anthropy = 0.5
num_core = cpu_count()
max_real_time = 3600 * 1


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
def angle_differences(x1, y1, goal):
    angle1 = math.atan2(y1, x1)
    raw_difference = math.degrees(goal - angle1)
    normalized_difference = (raw_difference + 180) % 360 - 180
    return normalized_difference


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

    '''Simulation loop'''
    i = 0
    while True:
        # Heading direction input to Delta7s
        neuron_df[i,0:16] = sinusoid_shape(heading_direction(agent_list[i][2]),1)
        # Goal direction input to PFLs (two goals)
        neuron_df[i,16:32] = sinusoid_shape(small_direction,small_ratio)
        neuron_df[i,32:48] = sinusoid_shape(high_direction,high_ratio)
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
        orientation = update_orientation(agent_list[i][0], rotation, noise_deviation=agent_noise)
        # Update agent's position
        x,y = update_position(agent_list[i][0], agent_list[i][1], orientation)
        # Update agent's dataframe
        agent_list.append([x,y,orientation])
        # check if the agent has reached the end
        if euclidean_distance(0,0,x,y) >= simulation_border:
            output = angle_differences(x,y, math.radians(45 * (high_direction-1)))
            break
        if i > 500:
            output = 180
            break
        i += 1

    return output

## Mutation function
def mutation(individual_matrix, probability_connectivity, probability_neuron):
    rows = individual_matrix.shape[0] - 16 - 16
    columns = individual_matrix.shape[1] - 48
    # Change connectivity
    mask = np.random.rand(rows,columns) < probability_connectivity
    random_values = np.random.uniform(-1, 1, size=(rows,columns))
    individual_matrix[16:-16,48:] = np.where(mask, random_values, individual_matrix[16:-16,48:])
    # Add/remove neurons
    if np.random.rand() < probability_neuron:
        decision = np.random.rand()
        if individual_matrix.shape[0] == 64:
            decision = 0
            neuron_index = 48
        else:
            neuron_index = np.random.randint(48, individual_matrix.shape[0]-16)
        if decision < 0.5: # Add neuron
            row_neuron = np.zeros((1,individual_matrix.shape[1]))
            row_neuron[0,48:] = np.random.uniform(-1, 1, size=(1,columns))
            individual_matrix = np.insert(individual_matrix, neuron_index, row_neuron, axis=0)
            col_neuron = np.zeros((individual_matrix.shape[0],1))
            col_neuron[16:-17,0] = np.random.uniform(-1, 1, size=(rows,1)).flatten()
            individual_matrix = np.hstack((individual_matrix[:,:neuron_index], col_neuron,individual_matrix[:,neuron_index:]))
        elif individual_matrix.shape[0] > 64: # Remove neuron
            individual_matrix = np.delete(individual_matrix, neuron_index, axis=0)
            individual_matrix = np.delete(individual_matrix, neuron_index, axis=1)
    return individual_matrix

## First generation function
def first_generation(nb_individuals):
    # Import connectivity matrix
    path = "Optimization_algorithm\Evolutionary_matrix.xlsx"
    matrix = pd.read_excel(path, header=None)
    matrix = matrix.to_numpy()
    # Create all randomly generated individuals
    ind = []
    for _ in range(nb_individuals):
        nb_neurons = random.randint(1, 16)
        added_rows = np.zeros((nb_neurons, matrix.shape[1]))
        added_rows[:, -16:] = np.random.uniform(-1, 1, size=(nb_neurons,16))
        ind_matrix = np.vstack((matrix[:48 ,:], added_rows, matrix[48:, :]))
        added_cols = np.zeros((ind_matrix.shape[0], nb_neurons))
        added_cols[16:-16, :] = np.random.uniform(-1, 1, size=(16*2 + nb_neurons,nb_neurons))
        ind_matrix = np.hstack((ind_matrix[:, :48], added_cols, ind_matrix[:, 48:]))
        ind.append(ind_matrix)
    return ind

## Clonal reproduction function
def clonal_reproduction(matrix_list, score_list):
    # Sort results
    score_list = [180-score for score in score_list]
    listed = list(zip(matrix_list, score_list))
    listed.sort(key=lambda x: x[1])
    # Split the list
    winners = listed[len(listed)//2:]
    # Make the winners reproduce
    total_score = sum(score for _,score in winners)
    if total_score == 0:
        return matrix_list
    clones = []
    for i in range(len(winners)):
        proportion = (round(winners[i][1]/total_score * (len(listed)//2)))
        for _ in range(proportion+1):
            clones.append(winners[i][0])
    return clones


### ----- Evolutionary algorithm
def run_algorithm(nb_individuals, nb_generations, connectivity_anthropy, neuron_anthropy, num_core, max_time):
    
    '''Initialisation'''
    ind_list = first_generation(nb_individuals)
    start_time = time.time()
    start_time_dt = datetime.fromtimestamp(start_time)
    print("===== Simulation starting... =====")
    print("Start time:", start_time_dt.strftime("%Y-%m-%d %H:%M:%S"))
    end_time = start_time + max_time
    end_time_dt = datetime.fromtimestamp(end_time)
    print("End time:", end_time_dt.strftime("%Y-%m-%d %H:%M:%S"))

    '''Graphical output'''
    mean_list = []
    variance_list = []
    min_max_list = []
    fig, ax = plt.subplots()
    plt.ion()
    ax.set_xlabel('Generation')
    ax.set_ylabel('Score')

    # Generations loop
    for gen in tqdm(range(nb_generations), desc="Processing", unit="iteration"):

        '''Simulations'''
        with Pool(num_core) as pool:
            score_list = pool.map(fitness_function, ind_list)

        '''Update graphical output'''
        # Calculate statistics
        mean_score = np.mean(score_list)
        variance_score = np.var(score_list)
        min_max_score = (min(score_list), max(score_list))
        # Update lists
        mean_list.append(mean_score)
        variance_list.append(variance_score)
        min_max_list.append(min_max_score)
        # Update plot
        ax.clear()
        ax.plot(range(gen + 1), mean_list, label='Mean')
        ax.fill_between(range(gen + 1), np.array(mean_list) - np.array(variance_list), np.array(mean_list) + np.array(variance_list), alpha=0.3, label='Variance')
        ax.plot(range(gen + 1), [x[0] for x in min_max_list], label='Min')
        ax.plot(range(gen + 1), [x[1] for x in min_max_list], label='Max')
        ax.legend()
        plt.pause(0.001)


        '''Save best matrix'''
        path = "Optimization_algorithm\Evolutionary_results\Best_contestant_gen" + str(gen) + "_score" + str(max(score_list)) + ".xlsx"
        best = ind_list[score_list.index(max(score_list))]
        best = pd.DataFrame(best)
        best.to_excel(path)

        '''Selection and reproduction'''
        ind_list = clonal_reproduction(ind_list, score_list)

        '''Mutation'''
        ind_list = [mutation(ind, connectivity_anthropy, neuron_anthropy) for ind in ind_list]

        '''Check timer'''
        if time.time() - start_time > max_time:
            break

    print("===== End of the simulation! =) =====")

    plt.ioff()  # Turn off interactive mode
    plt.show()


### ----- Run algorithm
if __name__ == "__main__":
    run_algorithm(nb_individuals, nb_generations, connectivity_anthropy, neuron_anthropy, num_core, max_real_time)