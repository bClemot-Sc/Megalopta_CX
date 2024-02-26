#### Script for optimization of compass circuit gain values
## Autor: Bastien Clémot


## ----- Import packages
import csv
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import time
start_time = time.time()
from CX_Script import initialise_dataframes, linear_activation, CIU_activation, update_orientation


## ----- connectivity matrix and IDs list
CON_MAT = np.genfromtxt("Connectivity_pattern.csv", delimiter=',')
with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))


## ----- Get IDs index
IND_PEN = [i for i, element in enumerate(COL_IDS) if "PEN" in element]
IND_EPG = [i for i, element in enumerate(COL_IDS) if "EPG" in element]
IND_PEG = [i for i, element in enumerate(COL_IDS) if "PEG" in element]
IND_TR = [i for i, element in enumerate(COL_IDS) if "TR" in element]
IND_D7 = [i for i, element in enumerate(COL_IDS) if "d7-" in element]
IND_CIU = [i for i, element in enumerate(COL_IDS) if "CIU" in element]


## ----- Adjustable rotation adressing function
def compare_headings_v2(previous_heading, new_heading, gain):
    TRr = 0
    TRl = 0
    heading_difference = (new_heading - previous_heading) % 360
    if heading_difference == 0:
        pass
    elif heading_difference <= 180:
        TRl = gain
    else:
        TRr = gain
    return TRl, TRr


## ----- Model function
def model(params):
    epg_to_pen, pen_to_epg, tr_to_pen, epg_to_peg, peg_to_epg, epg_to_d7, d7_to_peg, d7_to_pen, d7_to_d7 = params

    # Create matrix with the new gains
    ALT_MAT = np.copy(CON_MAT)
    ALT_MAT[np.ix_(IND_PEN,IND_EPG)] = ALT_MAT[np.ix_(IND_PEN,IND_EPG)] * epg_to_pen                               
    ALT_MAT[np.ix_(IND_EPG,IND_PEN)] = ALT_MAT[np.ix_(IND_EPG,IND_PEN)] * pen_to_epg
    ALT_MAT[np.ix_(IND_PEG,IND_EPG)] = ALT_MAT[np.ix_(IND_PEG,IND_EPG)] * epg_to_peg
    ALT_MAT[np.ix_(IND_EPG,IND_PEG)] = ALT_MAT[np.ix_(IND_EPG,IND_PEG)] * peg_to_epg
    ALT_MAT[np.ix_(IND_D7,IND_EPG)] = ALT_MAT[np.ix_(IND_D7,IND_EPG)] * epg_to_d7
    ALT_MAT[np.ix_(IND_PEG,IND_D7)] = ALT_MAT[np.ix_(IND_PEG,IND_D7)] * d7_to_peg
    ALT_MAT[np.ix_(IND_PEN,IND_D7)] = ALT_MAT[np.ix_(IND_PEN,IND_D7)] * d7_to_pen
    ALT_MAT[np.ix_(IND_D7,IND_D7)] = ALT_MAT[np.ix_(IND_D7,IND_D7)] * d7_to_d7

    # Select compass connectivity matrix only
    index = IND_PEN + IND_EPG + IND_PEG + IND_TR + IND_D7 + IND_CIU
    SUB_MAT = ALT_MAT[np.ix_(index, index)]

    # Initialisation
    SIM_TIME = 200
    Df, Act_df = initialise_dataframes(COL_IDS,SIM_TIME)
    Act = Act_df.iloc[:,index]
    Act = Act.copy()
    expected_EPG = pd.DataFrame(0.0, index=range(SIM_TIME+1), columns=range(16))

    # Time loop
    for i in range(SIM_TIME):

        # Update CIU activity input
        if i<(SIM_TIME/2):
            Act.loc[i, "CIU" + CIU_activation(Df.loc[i, "Orientation"])] = 1
                                        
        # Update TR activity input
        if i==0:
            pass
        else:
            Act.loc[i,"TRl"], Act.loc[i,"TRr"] = compare_headings_v2(Df.loc[i-1, "Orientation"], Df.loc[i, "Orientation"], tr_to_pen)

        # Save real orientation
        real_orientation = [0] * 8
        real_orientation[int(CIU_activation(Df.loc[i, "Orientation"]))-1] = 1
        expected_EPG.iloc[i] = real_orientation * 2

        # Update activity vector with connectivity matrix
        Act.iloc[i+1] = linear_activation(np.dot(SUB_MAT,Act.iloc[i]))

        # Update orientation
        Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],0, 0.4)

    # Compare simulation with expected EPG
    sim_err = 0
    for j in range(int(SIM_TIME/2)):
        epg_sim = np.array(Act.iloc[j, Act.columns.get_loc("EPG1"):Act.columns.get_loc("EPG16")+1])
        epg_exp = np.array(expected_EPG.iloc[j])
        sim_err += np.sum(np.abs(epg_sim - epg_exp))
            
    return sim_err


## ----- Define parameters 
param_constraints = (
    (0, 0),  # epg_to_pen
    (0, 0),  # pen_to_epg
    (0, 0),  # tr_to_pen
    (0.1, 1.0),  # epg_to_peg
    (0.1, 1.0),  # peg_to_epg
    (0.1, 1.0),  # epg_to_d7
    (0.1, 1.0),  # d7_to_peg
    (0, 0),  # d7_to_pen
    (0.1, 1.0)  # d7_to_d7
)


## Multi start optimization algorithm
best_result = None
for _ in range(10):
    # Initial conditions (random)
    initial_params = [np.random.uniform(low, high) for low, high in param_constraints]
    # Optimization algorithm
    result = minimize(model, initial_params, method='Powell', bounds=param_constraints)
    if best_result is None or result.fun < best_result.fun:
        best_result = result


## ----- Final output
# Afficher les résultats
print("Best gain values:", best_result.x)
print("Minimal error:", best_result.fun)
print("Message:", best_result.message)
print("Success:", best_result.success)
end_time = time.time()
elapsed_time = end_time - start_time
print("Runing time:", elapsed_time)
