# Projet Python M2 - 2026

## But du projet

Le but de ce projet est de simuler un problème à $N$ corps mûs par la gravité, ici une galaxie.

## Prérequis

Pour mener à bien se projet, les modules suivants seront nécessaires :
  - numpy (numerical python), indispensable pour le calcul numérique !
  - sdl2 : Couche fine de la bibliothèque SDL 2 (Simple Direct Layer) pour l'affichage
  - OpenGL : module python pour l'accélération graphique 3D
  - numba  : JIT (Just In Time) compilateur pour du code python. Permet des gains de performance significatifs et peut permettre l'usage d'un GPGPU (Cuda, donc NVidia uniquement).

Création d'un environnement Python
```
python3 -m venv ./venv
source ./venv/bin/activate
```

```
brew install sdl2
pip install PyOpenGL
pip install PySDL2
pip install numba
```

### Définition d'une galaxie

La script ```galaxy_generator.py```permet de créer des données permettant de simuler une galaxie avec $N$ étoile et un trou noir central.

**Invocation** :

```python3 galaxy_generator.py <nombre étoiles> <nom du fichier à créer>```

### But du projet

Le but du projet est de simuler la dynamique d'une galaxie au travers d'un problème à $N$ corps.

Nous allons définir une étoile $i$ par :
  - Sa masse $m_{i}$ exprimée en masse solaire
  - Sa position $\vec{p}_{i}$
  - Sa vitesse $\vec{v}_{i}$
  - Sa couleur RGB $c_{i}$

La couleur de l'étoile est déterminée en fonction de sa masse :
  - Si $m_{i}>5$, alors l'étoile sera de couleur bleu-blanc (géante bleue : *RGB=(150,180,255)*)
  - Si $m_{i}>2$, alors l'étoile sera de couleur blanche (*RGB=(255,255,255)*)
  - Si $m_{i}\geq 1$ alors l'étoile sera jaune (étoile type soleil : *RGB=(255,255,200)*)
  - Sinon c'est une étoile naine rouge (*RGB=(250,150,100)*)

On rajoute de plus un trou noir de couleur noire (*RGB=(0,0,0)*) bien plus massif que les étoiles de la galaxie.

La position et la vitesse de l'étoile (ou du trou noir) seront mis à jour à chaque pas de temps, tandis que sa couleur et sa masse resteront constantes durant toute la simulation.

Pour mettre à jour la vitesse et la position, on calcule en premier lieu l'accélération subie pour chaque étoile $i$ en utilisant les lois de Newton :

$$
\left\lbrace
\begin{array}{lcl}
\vec{f}_{i}(t) & = & m_{i}.\vec{a}_{i}(t) \\
\vec{f}_{i}(t) & = & \displaystyle \sum_{j\neq i} \mathcal{G}\frac{m_{i}m_{j}}{\|\vec{p}_{j}(t)-\vec{p}_{i}(t)\|^{3}}(\vec{p}_{j}(t)-\vec{p}_{i}(t))
\end{array}
\right.
$$

où $\mathcal{G} = 1.560339.10^{-13}$. Les unités utilisées sont :
  - en année lumière pour les distances
  - en masse solaire pour les masses
  - en année terrestre pour la durée

Puis on met à jour la position et la vitesse :

$$
\left\lbrace
\begin{array}{lcl}
\vec{v}_{i}(t+\delta_{t}) & = & \vec{v}_{i}(t) + \delta t \vec{a}_{i}(t) \\
\vec{p}_{i}(t+\delta_{t}) & = & \vec{p}_{i}(t)+\delta t \vec{v}_{i}(t) + \frac{1}{2}(\delta t)^{2}\vec{a}_{i}(t)
\end{array}
\right.
$$

### Première version : programmation naïve

Dans un premier temps, on va simuler la galaxie par une approche objet. On va donc créer :
  - Une classe ```Corps``` dont chaque instance contiendra la masse, la couleur et la position d'un corps, une méthode qui met à jour la position et la vitesse par rapport à une accélération et un pas de temps $\delta t$, une méthode calculant la distance entre le corps et un autre corps (on suppose que les corps sont réduits à un point);
  
  - Une classe ```NCorps``` qui contiendra une collection de corps. On pourra calculer la force d'attraction du corps appelant la méthode par les autres corps contenus dans ```NCorps```.

Pour visualiser à chaque pas de temps la position des corps dans la galaxie, on utilisera ```visualizer3d_vbo.py``` si votre carte graphique supporte les ```vertices buffer object``` et si vous n'avez pas de rendu, on utilisera ```visualizer3d_sans_vbo.py```
(qui prend plus de temps pour l'affichage mais peut suffire pour le projet).

Mesurer le temps de calcul et le nombre de frame par secondes de votre code en fonction du nombre de corps présents dans la simulation.

### Deuxième version

Afin d'accélérer le temps de calcul, on va vectoriser les calculs gourmand en temps de calcul en modifiant l'organisation des données en mémoire :
  - un tableau contenant la position de tous les corps de la galaxie
  - un tableau contenant la vitesse de tous les corps de la galaxie
  - Un tableau contenant les couleurs des différents corps de la galaxie.

Un corps sera défini par son indice dans ses trois tableaux. 

  - Modifiez le programme afin au maximum de vectoriser le code à l'aide de ses trois tableaux.

Comparez les temps de calcul pour diverses galaxies (avec un nombre d'étoiles variable) entre la première et deuxième version.

### Troisième version : utilisation de numba

Afin d'accélérer le calcul des forces pour chaque corps contenu dans la galaxie, nous allons utiliser ```numba```.
  - Mesurez le temps pris par les diverses fonctions
  - Utilisez le décorateur ```njit``` afin d'optimiser les routines critiques en temps de calcul
  - Mesurez le gain de temps obtenu à l'aide de numba
  - Paralléliser si possible les boucles en rajoutant l'option ```parallel=True``` et en utilisant la fonction ```prange``` proposée par numba. 
  - Mesurez de nouveau le temps obtenu et comparez en fonction du nombre de cœurs possédés par la plateforme sur laquelle vous exécutez le code.

Essayez plusieurs pas de temps (en année terrestre). Que constatez-vous ? Comment expliquez-vous ce phénomène ?

