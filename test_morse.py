import numpy as np

# Paramètres de la grille
square_size = (5.0, 5.0)  # 100/20 = 5

# Tableau de positions (N, 2) : 10 étoiles avec coordonnées (x, y)
position = np.array([
    [12.5, 45.2], [ 3.1,  8.9], [98.2, 99.1], [50.0, 50.0], [14.9, 42.1],
    [ 1.0,  1.0], [85.5, 12.3], [44.4, 46.7], [22.2, 88.8], [99.9,  0.1]
])



def grid_matrice_crs(position): 
    """
    objectif : avoir 2 listes : une qui compte le combre d'etoiles qu'il y a dans les cases d'avant pour chaque case 
    une qui organise les etoiles dans l'ordre des cases
    position est la liste des positions des etoiles, position [id]][0] est la position_x de l'etoile id 

    """

    global square_size

    aux = np.zeros(400, dtype=int) #20x20=400 cases 
    beg_cases = np.zeros(401, dtype=int) #20x20=400 cases
    for i in range(len(position)): 
        indice_colonne = min(int(position[i][0]/square_size[0]),19) #min pour eviter les erreurs d'indice si une etoile se trouve a la limite de la grille
        indice_ligne = min(int(position[i][1]/square_size[1]),19)
        place = indice_ligne*20 + indice_colonne #ligne_indice*20 pour compter le nombre de cases dans les lignes au dessus et + indice_colonne pour decaler l'etoile dans la bonne colonne de la ligne 
        aux[place] += 1

    beg_cases[1:] = np.cumsum(aux)

    aux2 = np.zeros(400, dtype=int) 
    tab = np.zeros(len(position), dtype=int)
    for i in range(len(position)): 
        indice_colonne = min(int(position[i][0]/square_size[0]),19)
        indice_ligne = min(int(position[i][1]/square_size[1]),19)
        place = indice_ligne*20 + indice_colonne
        if place == 0 : 
            place_finale = aux2[place]
        else : 
            place_finale = aux2[place] + beg_cases[place]
        tab[place_finale] = i
        aux2[place] += 1


    return beg_cases, tab


grid_matrice_crs(position)