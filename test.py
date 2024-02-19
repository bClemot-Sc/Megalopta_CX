import numpy as np
matrice_originale = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
nouvelle_matrice = np.copy(matrice_originale)
nouvelle_matrice[0, 0] = 100
print("Matrice d'origine :\n", matrice_originale)
print("Nouvelle matrice :\n", nouvelle_matrice)
