import pandas as pd
import csv

with open("Neurons_IDs.csv", "r") as file:
        COL_IDS = next(csv.reader(file, delimiter=','))

print(len(COL_IDS))