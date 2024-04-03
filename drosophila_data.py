#### Script for drosophila adjacency matrix
## Autor: Bastien Clémot


##### ----- Import packages
from fafbseg import flywire


##### ----- Set global Flywire parameters
# Set default dataset
flywire.set_default_dataset("public")
# Check materialisation
flywire.get_materialization_versions()