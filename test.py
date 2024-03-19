import pandas as pd

# Créez un exemple de DataFrame
data = {'colonne_X': [0, 0, 0, 0, 0, 0, 0, 0, 0]}
df = pd.DataFrame(data)

# Trouvez l'indice de la première occurrence où la colonne passe de 0 à 1
indice_ligne = (df['colonne_X'] == 1).idxmax()

# Affichez le résultat
print("L'indice de la première occurrence où la colonne passe de 0 à 1 :", indice_ligne)


print(int(3.3))