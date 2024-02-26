#### Script for optimization of compass circuit gain values
## Autor: Bastien Clémot


## ----- Import packages
import csv
import math
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import time
start_time = time.time()
from CX_Script import initialise_dataframes, linear_activation, CIU_activation, update_orientation, update_position


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
IND_PFN = [i for i, element in enumerate(COL_IDS) if "PFN" in element]
IND_PFL = [i for i, element in enumerate(COL_IDS) if "PFL" in element]
IND_HD = [i for i, element in enumerate(COL_IDS) if "hd" in element]
IND_TS = [i for i, element in enumerate(COL_IDS) if "TS" in element]

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
    epg_to_pen, pen_to_epg, tr_to_pen, epg_to_peg, peg_to_epg, epg_to_d7, d7_to_peg, d7_to_pen, d7_to_d7, d7_to_pfn, d7_to_pfl, pfn_to_pfl, pfn_to_hd, hd_to_hd, hd_to_pfl = params

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
    ALT_MAT[np.ix_(IND_PFN,IND_D7)] = ALT_MAT[np.ix_(IND_PFN,IND_D7)] * d7_to_pfn
    ALT_MAT[np.ix_(IND_PFL,IND_D7)] = ALT_MAT[np.ix_(IND_PFL,IND_D7)] * d7_to_pfl
    ALT_MAT[np.ix_(IND_PFL,IND_PFN)] = ALT_MAT[np.ix_(IND_PFL,IND_PFN)] * pfn_to_pfl
    ALT_MAT[np.ix_(IND_HD,IND_PFN)] = ALT_MAT[np.ix_(IND_HD,IND_PFN)] * pfn_to_hd
    ALT_MAT[np.ix_(IND_HD,IND_HD)] = ALT_MAT[np.ix_(IND_HD,IND_HD)] * hd_to_hd
    ALT_MAT[np.ix_(IND_PFL,IND_HD)] = ALT_MAT[np.ix_(IND_PFL,IND_HD)] * hd_to_pfl

    # Create an exploration matrix with no hd to PFL connection
    EXP_MAT = np.copy(ALT_MAT)
    EXP_MAT[np.ix_(IND_PFL,IND_HD)] = 0

    # Select target function only
    index = IND_PEN + IND_EPG + IND_PEG + IND_TR + IND_D7 + IND_CIU + IND_PFN + IND_PFL + IND_HD + IND_TS
    SUB_MAT1 = ALT_MAT[np.ix_(index, index)]
    SUB_MAT2 = EXP_MAT[np.ix_(index, index)]

    # Initialisation
    SIM_TIME = 100
    Df, Act_df = initialise_dataframes(COL_IDS,SIM_TIME)
    Act = Act_df.iloc[:,index]
    Act = Act.copy()
    nest_distance = pd.DataFrame({"distance" : [0] * (SIM_TIME+1)})

    # Time loop
    for i in range(SIM_TIME):

        # Update CIU activity input
        Act.loc[i, "CIU" + CIU_activation(Df.loc[i, "Orientation"])] = 1
                                        
        # Update TR activity input
        if i==0:
            pass
        else:
            Act.loc[i,"TRl"], Act.loc[i,"TRr"] = compare_headings_v2(Df.loc[i-1, "Orientation"], Df.loc[i, "Orientation"], tr_to_pen)

        # Save distance to nest (0,0)
        nest_distance.loc[i,"distance"] = int(math.hypot(0,0,Df.loc[i,"X"], Df.loc[i,"Y"]))

        # Activate homing vector halfway through the simulation
        if (i < SIM_TIME/2):

            # Update new activity with no hd → PFL (Alternative connectivity matrix)
            Act.iloc[i+1] = linear_activation(np.dot(SUB_MAT2, Act.iloc[i]))

        else:

            # Update new activity with complete connectivity matrix
            Act.iloc[i+1] = linear_activation(np.dot(SUB_MAT1, Act.iloc[i]))

        # Update rotational speed from PFL neurons
        Df.loc[i+1,"Rotation"] = Act.iloc[i+1, Act.columns.get_loc("PFL1"):Act.columns.get_loc("PFL8") + 1].sum() - Act.iloc[i+1, Act.columns.get_loc("PFL9"):Act.columns.get_loc("PFL16") + 1].sum()

        # Update orientation
        Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],0, 0.1)
        new_x, new_y = update_position(Df.loc[i,"X"],Df.loc[i,"Y"],Df.loc[i,"Speed"],Df.loc[i+1,"Orientation"])
        Df.loc[i+1, "X"] = new_x
        Df.loc[i+1, "Y"] = new_y

    # Compare simulation with expected direction
    sim_err = nest_distance.iloc[int(SIM_TIME/2):,0].sum()
            
    return sim_err


## ----- Define parameters 
param_constraints = (
    (0, 0),  # epg_to_pen
    (0, 0),  # pen_to_epg
    (0, 0),  # tr_to_pen
    (0.22, 0.22),  # epg_to_peg
    (0.83, 0.83),  # peg_to_epg
    (0.44, 0.44),  # epg_to_d7
    (0.66, 0.66),  # d7_to_peg
    (0, 0),  # d7_to_pen
    (0.66, 0.66),  # d7_to_d7
    (0.1, 1.0),  # d7_to_pfn
    (0.1, 1.0),  # d7_to_pfl
    (0.1, 1.0),  # pfn_to_pfl
    (0.1, 1.0),  # pfn_to_hd
    (0.1, 1.0),  # hd_to_hd
    (0.1, 1.0)  # hd_to_pfl
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
