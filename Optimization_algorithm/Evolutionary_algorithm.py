#### Script for evolutionary algorithm of multiple goal integration
## Author: Bastien Clémot

### ----- Import packages
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pandas as pd
import random
import time
from tqdm import tqdm


### ----- Quick access to some parameters
agent_noise = 10
simulation_border = 200
nb_individuals = 100
nb_generations = 1_000_000
connectivity_anthropy = 0.2
neuron_anthropy = 0.1
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

## Check if an angle value is between two others
def angle_between(x,y,border1,border2):
    # Get agent angle
    angle = math.atan2(y, x) * 180 / math.pi
    angle = (angle + 360) % 360
    # Normalise
    border1 = border1 % 360
    border2 = border2 % 360
    # Check pathways
    clockwise_distance = (border2 - border1) % 360
    counterclockwise_distance = (border1 - border2) % 360
    clockwise1 = (angle - border1) % 360
    counterclockwise1 = (border1 - angle) % 360
    clockwise2 = (border2 - angle) % 360
    counterclockwise2 = (angle - border2) % 360
    if clockwise_distance <= counterclockwise_distance:
        return clockwise1 <= clockwise_distance and clockwise2 <= clockwise_distance
    else:
        return counterclockwise1 <= counterclockwise_distance and counterclockwise2 <= counterclockwise_distance

### ----- Define evolutionary functions

## Fitness function
def fitness_function(individual_matrix):
    
    score = []
    
    for _ in range(3):

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
        bad_direction = direction1 if ratio1 < 0.5 else direction2

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
            # Score the agent's direction (weighted areas)
            if angle_between(x, y, 45*(direction1-1), 45*(direction2-1)):
                multiplier = 0.8
            else:
                multiplier = 0.2
            if angle_differences(x,y, math.radians(45 * (best_direction-1))) < 10:
                multiplier = 1
            elif angle_differences(x,y, math.radians(45 * (bad_direction-1))) < 10:
                multiplier = 0.5    
            direction_reward = (180 - angle_differences(x,y, math.radians(45 * (best_direction-1)))) * multiplier
                                  
            # check if the agent has reached the end
            if euclidean_distance(0,0,x,y) >= simulation_border:
                distance_reward = simulation_border 
                time_reward = 500 - i
                score.append(distance_reward + direction_reward + time_reward)
                break
            if i > 500:
                distance_reward = euclidean_distance(0,0,x,y)
                time_reward = 0
                score.append(distance_reward + direction_reward + time_reward)
                break
            i += 1

    return np.mean(score)

## Mutation function
def mutation(individual_matrix, probability_connectivity, probability_neuron):
    rows = individual_matrix.shape[0] - 16 - 32 - 16 
    columns = individual_matrix.shape[1] - 16 - 32 - 32
    # Change connectivity
    mask = np.random.rand(rows, columns) < probability_connectivity
    percent_change = np.random.uniform(-0.1, 0.1, size=(rows, columns))
    individual_matrix[48:-16, 80:] = np.where(mask, individual_matrix[48:-16, 80:] + percent_change, individual_matrix[48:-16, 80:])
    # Add/remove neurons
    if np.random.rand() < probability_neuron:
        decision = np.random.rand()
        if individual_matrix.shape[0] == 96:
            decision = 0
            neuron_index = 80
        else:
            neuron_index = np.random.randint(80, individual_matrix.shape[0]-16)
        if decision < 1: # Add neuron
            rows, columns = individual_matrix.shape
            # Insert row
            row_neuron = np.zeros((1, columns))
            non_zero_indices_row = np.random.choice(range(80, columns), int((columns - 80) * probability_connectivity), replace=False)
            row_neuron[0, non_zero_indices_row] = np.random.uniform(-1, 1, size=(1, len(non_zero_indices_row)))
            individual_matrix = np.vstack((individual_matrix[:neuron_index, :], row_neuron, individual_matrix[neuron_index:, :]))
            # Insert column
            col_neuron = np.zeros((rows + 1, 1))
            non_zero_indices_col = np.random.choice(range(48, rows - 16), int((rows - 64) * probability_connectivity), replace=False)
            col_neuron[non_zero_indices_col, 0] = np.random.uniform(-1, 1, size=(len(non_zero_indices_col), 1)).flatten()
            individual_matrix = np.hstack((individual_matrix[:, :neuron_index], col_neuron, individual_matrix[:, neuron_index:]))
        elif individual_matrix.shape[0] > 96: # Remove neuron
            individual_matrix = np.delete(individual_matrix, neuron_index, axis=0)
            individual_matrix = np.delete(individual_matrix, neuron_index, axis=1)
    return individual_matrix

## First generation function
def first_generation(nb_individuals):
    # Import connectivity matrix
    path = "Evolutionary_matrix.xlsx"
    matrix = pd.read_excel(path, header=None)
    matrix = matrix.to_numpy()
    # Create all randomly generated individuals
    ind = []
    for _ in range(nb_individuals):
        ind_matrix = np.copy(matrix)
        if random.random() < 0.5: # Remove PFN-PFL pathway
            ind_matrix[48:80, -16:] = 0
        if random.random() < 0.5: # Add new neurons
            nb_neurons = 16
            added_rows = np.zeros((nb_neurons, matrix.shape[1]))
            # added_rows[:, -16:] = np.random.uniform(-1, 1, size=(nb_neurons,16))
            ind_matrix = np.vstack((matrix[:80 ,:], added_rows, matrix[80:, :]))
            added_cols = np.zeros((ind_matrix.shape[0], nb_neurons))
            # added_cols[48:-16, :] = np.random.uniform(-1, 1, size=(16*2 + nb_neurons,nb_neurons))
            ind_matrix = np.hstack((ind_matrix[:, :80], added_cols, ind_matrix[:, 80:]))
        ind.append(ind_matrix)
    return ind

## Clonal reproduction function
def clonal_reproduction(matrix_list, score_list):
    # Sort results
    listed = list(zip(matrix_list, score_list))
    listed.sort(key=lambda x: x[1])
    # Take the 10 best 
    winners = listed[-10:]
    # Make the winners reproduce
    total_score = sum(score for _,score in winners)
    clones = []
    for i in range(len(winners)):
        clones.append(winners[i][0])
    for i in range(len(winners)):
        proportion = (round(winners[i][1]/total_score * (len(listed)-10)))
        for _ in range(proportion):
            clones.append(winners[i][0])
    clones = clones[:nb_individuals]
    return clones

## Cross-Over function
# def cross_over(matrix_list, score_list):
#    # Sort results 
#    listed = list(zip(matrix_list, score_list))
#    listed.sort(key=lambda x: x[1])
#    # Take the 10 best 
#    winners = listed[-10:]
#    parents = [winner for winner,_ in winners]
#    offsprings = parents
#    # Get reproductive rate for each winner
#    total_score = sum(score for _,score in winners)
#    rate = [round(score/total_score) for _,score in winners]
#    # Generate offsprings
#    while len(offsprings) < (matrix_list):
#        parent1 = random.choices(parents, rate)[0]
#        parent2 = random.choices(parents, rate)[0]
        

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
    std_list = []
    min_max_list = []
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()

    # Generations loop
    for gen in tqdm(range(nb_generations), desc="Processing", unit="iteration"):

        '''Simulations'''
        with Pool(num_core) as pool:
            score_list = pool.map(fitness_function, ind_list)
            
        print("Current best score:", max(score_list))

        '''Update graphical output'''
        # Calculate statistics
        mean_score = np.mean(score_list)
        std_score = np.std(score_list)
        min_max_score = (min(score_list), max(score_list))
        # Update lists
        mean_list.append(mean_score)
        std_list.append(std_score)
        min_max_list.append(min_max_score)
        # Update plot
        ax.clear()
        ax.plot(range(gen + 1), mean_list, label='Mean')
        ax.fill_between(range(gen + 1), np.array(mean_list) - np.array(std_list), np.array(mean_list) + np.array(std_list), alpha=0.3, label='Std')
        ax.plot(range(gen + 1), [x[0] for x in min_max_list], label='Min')
        ax.plot(range(gen + 1), [x[1] for x in min_max_list], label='Max')
        ax.legend()
        fig.canvas.draw()

        '''Save best matrix'''
        path = "EA_results/Best_contestant_gen" + str(gen) + "_score" + str(max(score_list)) + ".xlsx"
        best = ind_list[score_list.index(max(score_list))]
        best = pd.DataFrame(best)
        best.to_excel(path)

        '''Selection and reproduction'''
        ind_list = clonal_reproduction(ind_list, score_list)

        '''Mutation'''
        ind_list = ind_list[0:9] + [mutation(ind, connectivity_anthropy, neuron_anthropy) for ind in ind_list[10:]]

        '''Check timer'''
        if time.time() - start_time > max_time:
            break

    print("===== End of the simulation! =) =====")

    plt.ioff()  # Turn off interactive mode
    plt.show()


### ----- Run algorithm
if __name__ == "__main__":
    run_algorithm(nb_individuals, nb_generations, connectivity_anthropy, neuron_anthropy, num_core, max_real_time)