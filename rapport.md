# Projet Python : génération d'une galaxie

## Première version : programmation naïve

Dans cette première version, on utilise une approche objet avec la définition des classes *Corps* et *NCorps*. La classe *Corps* correspond à une étoile, caractérisée par sa masse, sa couleur, sa position et sa vitesse. La classe *NCorps* correspond elle à une collection d'objets de type Corps.  
On exécute le programme *galaxy_body.py* avec un pas de temps dt=0.01 afin d'assurer la précision et la stabilité du résultat.

Temps de calcul et nombre de frame par seconde en fonction du nombre des corps :

| Nombre de corps | 100 | 500 | 1000 | 2500 |
| --- | --- | --- | --- | --- |
| Temps de calcul | 1.42 s | 36.39 s | 143.05 s | 914.91 s |
| Nombre de frame par secondes | 5.63 | 0.27 | 0.07 | 0.01 |

## Deuxième version : vectorisation

Dans cette deuxième version, on vectorise les calculs afin d'accélérer le temps de calcul. Pour cela, on défini trois tableaux, contenant la position de tous les corps, la vitesse de tous les corps et la couleur de tous les corps. Un corps sera maintenant défini par son indice dans ces trois tableaux. On exécute le code *galaxy_vectorized.py* avec dt=0.01 .

Temps de calcul et nombre de frame par seconde en fonction du nombre des corps :

| Nombre de corps | 100 | 500 | 1000 | 2500 |
| --- | --- | --- | --- | ---- |
| Temps de calcul | 0.0168 s | 0.4740 s | 1.8783 s | 11.2770 s |
| Nombre de frame par secondes | 25 | 11 | 4 | 0.8 |

On remarque que la version vectorisée est beaucoup plus rapide (presque 100x plus) que la première version avec les classes. Cela nous permet donc de générer des galaxies avec un nombre d'étoiles important tout en ayant un temps de calcul convenable.

## Troisième version : utilisation de numba

On va maintenant utiliser numba qui va permettre des gains de performance significatifs par rapport aux versions précédentes.
Pour comparer cela, on mesure le temps d'exécution des fonctions *calculate_acceleration*, *step* et *load_galaxy* sans utiliser numba. On réalise le calcul pour un pas de temps 0.01 et 2500 corps. Les résultats sont :

- *calculate_acceleration* : 73.8397 s
- *step* : 73.1246 s
- *load_galaxy* : 0.0200 s

Les temps étant très élevés, on ajoute numba avec le décorateur *njit* qui permet d'optimiser les routines ayant un temps de calcul important. On prends encore un pas de temps dt=0.01 et 2500 corps. Les résultats deviennent :

- *calculate_acceleration* : 1.4749 s
- *step* : 1.5534 s
- *load_galaxy* : 0.0576 s

Il y a donc une réduction très significative du temps de calcul pour ces différentes fonctions ce qui rend l'exécution totale du code bien plus rapide et permet de générer des galaxies avec un nombre beaucoup plus élevé d'étoiles.

Pour avoir encore mieux, il est possible de paralléliser le code tout en utilisant numba avec le ligne *@numba.njit(parallel=True)* et la fonction *prange*. On mesure à nouveau le temps de chaque fonctions avec dt=0.01 et 2500 corps. Le nombre de coeurs peut aussi être modifié avec *$env:NUMBA_NUM_THREADS=8* par exemple pour 8 coeurs.

| Nombre de coeurs        | 4        | 8        | 16       |
|-------------------------|----------|----------|----------|
| *calculate_acceleration*  | 0.2841 s | 0.1705 s | 0.1487 s |
| *step*                    | 0.2453 s | 0.1713 s | 0.0889 s |
| *load_galaxy*             | 0.0552 s | 0.0875 s | 0.0709 s |

Pour 4 coeurs seulement, il y a déjà un réduction importante du temps de calcul par rapport à la version non parallèle. En ajoutant des coeurs, les résulats sont encore plus rapides, cependant il n'est pas nécessaire d'en rajouter trop, surtout si le nombre de corps de la galaxie n'est pas si important.


Lorsqu'on essaye différents pas de temps, on remarque que pour les pas de temps trop grands (0.1 par exemple), la simulation est instable. Certaines planètes sortent complètement de la galaxie par exemple. Pour des pas de temps plus petit, on obtient des résultats plus cohérents, les mouvements sont stables et les orbites plus réalistes. Cette différence s'explique par le fait que la méthode d'Euler est instable pour des pas de temps trop grand, l'erreur locale s'accumule à chaque itération ce qui rend la simulation fausse physiquement.

## Quatrième version : Verlet

Dans cette version, on crée une grille et on assigne chaque étoile au morceau de la grille auquelle elle appartient. Pour chaque cellule d'une grille, on calcule son centre de gravité. L'idée de cette méthode est de rendre plus rapide les calculs d'accélération si une étoile est trop éloigné de l'étoile qu'on étudie en remplacant un groupe d'étoiles lointaines par leur centre de masse. Le critère exact est *si 0.5 * dist > radius* , (avec *dist* la distance euclidienne entre l'étoile étudié et le centre de gravité et *radius* le rayon d'une cellule) alors on calcule l'accélération par rapport au centre de gravité de la cellule concernée. Si le critère n'est pas satisfait alors on calcule l'accélération comme dans les versions précédentes. En faisant cela, on réduit fortement la complexité de la fonction *calculate_accelerations* car chaque étoile n'interagit plus avec chacune des autres étoiles. Le temps d'exécution général du code est donc réduit ce qui permet de générer des galaxies de taille plus grande.

**donner valeurs temps**

## Cinquième version : Verlet avec Numba