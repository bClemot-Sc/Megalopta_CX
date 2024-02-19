#### Script for optimization of compass circuit gain values
## Autor: Bastien Clémot


## ----- Import packages
import csv
import numpy as np
import pandas as pd
from CX_Script import initialise_dataframes, linear_activation, CIU_activation, update_orientation


## ----- connectivity matrix and IDs list
CON_MAT = np.genfromtxt("Theorical_connectivity_matrix.csv", delimiter=',')
with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))


## ----- Get IDs index
IND_PEN = [i for i, element in enumerate(COL_IDS) if "PEN" in element]
IND_EPG = [i for i, element in enumerate(COL_IDS) if "EPG" in element]
IND_PEG = [i for i, element in enumerate(COL_IDS) if "PEG" in element]
IND_TR = [i for i, element in enumerate(COL_IDS) if "TR" in element]
IND_D7 = [i for i, element in enumerate(COL_IDS) if "d7-" in element]
IND_CIU = [i for i, element in enumerate(COL_IDS) if "CIU" in element]


## ----- Select compass connectivity matrix only
index = IND_PEN + IND_EPG + IND_PEG + IND_TR + IND_D7 + IND_CIU
SUB_MAT = CON_MAT[np.ix_(index, index)]


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


## ----- Run simulation optimization
# Initialize error 
err_tot = 10_000

# Range of gain values to be tested
for epg_to_pen in [float(a) / 100 for a in range(0, 105, 5)]:
    for pen_to_epg in [float(b) / 100 for b in range(0, 105, 5)]:
        for tr_to_pen in [float(c) / 100 for c in range(0, 105, 5)]:
            for epg_to_peg in [float(d) / 100 for d in range(0, 105, 5)]:
                for peg_to_epg in [float(e) / 100 for e in range(0, 105, 5)]:                    

                    # Create matrix with the new gains
                    ALT_MAT = np.copy(SUB_MAT)
                    for x in IND_PEN:
                        for y in IND_EPG:
                            ALT_MAT[x,y] = epg_to_pen
                            ALT_MAT[y,x] = pen_to_epg
                    for z in IND_PEG:
                        for w in IND_EPG:
                            ALT_MAT[z,w] = epg_to_peg
                            ALT_MAT[w,z] = peg_to_epg

                    # Initialisation
                    SIM_TIME = 100
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
                        Act.iloc[i+1] = linear_activation(np.dot(ALT_MAT,Act.iloc[i]))

                        # Update orientation
                        Df.loc[i+1, "Orientation"] = update_orientation(Df.loc[i,"Orientation"],0, 0.4)

                    # Compare simulation with expected EPG
                    sim_err = 0
                    for j in range(int(SIM_TIME/2), SIM_TIME+1):
                        epg_sim = np.array(Act.iloc[j, Act.columns.get_loc("EPG1"):Act.columns.get_loc("EPG16")+1])
                        epg_exp = np.array(expected_EPG.iloc[j])
                        sim_err += np.sum(np.abs(epg_sim - epg_exp))

                    # Save if better
                    if sim_err < err_tot:
                        res_epg_pen = epg_to_pen
                        res_pen_epg = pen_to_epg
                        res_tr_pen = tr_to_pen
                        res_epg_peg = epg_to_peg
                        res_peg_epg = peg_to_epg
                        err_tot = sim_err


## ----- Final output
print("• Gain EPG → PEN:")
print(res_epg_pen)
print("• Gain PEN → EPG:")
print(res_pen_epg)
print("• Gain TR → PEN:")
print(res_tr_pen)
print("• Gain EPG → PEG:")
print(res_epg_peg)
print("• Gain PEG → EPG:")
print(res_peg_epg)
print("• Total error:")
print(err_tot)



            

