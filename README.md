# Central complex model
#### Autor: Bastien Clémot


## Files
- **CX_Launcher.py :** The actual file that needs to be run. It includes the GUI and calls the CX script.
- **CX_Script.py :** File containing all the functions for the CX model to work. It also includs the simulation loop.
- **Theorical_connectivity_matrices.xlsx :** The file containing all the connectivity between neurons as well as their synaptic strength.
- **Eddit_matrix.py :** A script to run for converting the previous connectivity matrix file into readable files for the CX model script.


## CX_Script.py details
- **Import packages**
- **Import connectivity matrix and neuron IDs list**
- **Define functions :**
  - *initialise_dataframes(ids_list, time) :* Create dataframes to stock simulated datas for the neuron activity values and the stirring parameters of the agent.
  - *adjust_orientation(angle) :* Wrape the angle between 0 and 360°.
  - *matrix_multiplication(connectivity_matrix, activity_vector) :* Propagate neuron activation through connections.
  - *linear_activation(activity_vector) :* Threshold neuron activation between 0 and 1.
  - *logic_activcation(activity_vector, threshold) :* Binarise neuron activity based on a given threshold.
  - *CIU_activation(heading_direction) :* Gives the CIU corresponding to the current heading direction.
  - *compare_headings(previous_heading,new_heading) :* Gives the TR corresponding to the current turning direction.
  - *update_position(x,y,translational_speed, orientation) :* Update the coordinates of the agent based on the stiring outputs.
  - *update_orientation(orientation, rotational_speed, noise_factor) :* Update the heading of the agent based on the stiring outputs.
  - *activity_heatmap(activity_df) :* Plot the evolution of neurons activity through time.
  - *clean_ids(ids) :* Process neurons IDs for building the heatmap.
  - *plot_stirring(Df) :* Plot the path made by the agent during the simulation.
- **Run the simulation:**
  - Initialise the dataframes
  - Run the time loop:
    - Check if it's currently daytime:
      - Update CIU activity depending on the current heading
    - Update TS activity input
    - Update TR activity input
    - Update activity vector while applying the chosen activation function
    - Update position and orientation based on the new activity vector
  -  Plot heatmap and path 


## References
- Goulard, R., Heinze, S., & Webb, B. (2023). Emergent spatial goals in an integrative model of the insect central complex. PLOS Computational Biology, 19(12), e1011480.
- Stone, T., Webb, B., Adden, A., Weddig, N. B., Honkanen, A., Templin, R., ... & Heinze, S. (2017). An anatomically constrained model for path integration in the bee brain. Current Biology, 27(20), 3069-3085.