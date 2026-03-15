# HDBSCAN Package

Un package Python implémentant l'algorithme HDBSCAN pour le clustering hiérarchique de données.

## Description

Ce package fournit une implémentation complète de l'algorithme HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) avec visualisation intégrée. Il permet de détecter des clusters de formes arbitraires et gère automatiquement le bruit dans les données.

## Installation

### Prérequis

- Python 3.7+
- NumPy
- Matplotlib
- Scikit-learn

### Installation du package

```bash
# Cloner ou copier le répertoire du package
git clone <repository-url>
cd hdbscan_package

# Ou simplement copier les fichiers dans votre projet
```

## Structure du package

```
hdbscan_package/
├── __init__.py          # Exports des classes principales
├── distance.py          # Calcul des matrices de distance
├── hdbscan.py          # Classe principale HDBSCAN
├── mst.py              # Arbre couvrant minimal (MST)
├── union_find.py       # Structure Union-Find
├── linkage_tree.py     # Arbre de liaison hiérarchique
├── condensed_tree.py   # Arbre condensé
└── demo.py            # Exemple d'utilisation
```

## Utilisation rapide

### Importation

```python
from hdbscan_package import HDBSCAN
import numpy as np
from sklearn import datasets
```

### Exemple simple

```python
# Génération de données
moons, _ = datasets.make_moons(n_samples=50, noise=0.05)
blobs, _ = datasets.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
X = np.vstack([moons, blobs])

# Clustering HDBSCAN
hdbscan = HDBSCAN(min_pts=5, min_cluster_size=5)
hdbscan.fit(X)

# Obtention des labels avec sélection automatique du seuil
labels = hdbscan.get_labels(auto=True)

print(f"Clusters détectés: {len(set(labels[labels >= 0]))}")
print(f"Points de bruit: {np.sum(labels == -1)}")
```

### Utilisation avancée

```python
# Avec paramètres personnalisés
hdbscan = HDBSCAN(min_pts=3, min_cluster_size=10)
hdbscan.fit(X)

# Sélection manuelle du seuil
labels_percentile = hdbscan.get_labels(percentile=70)
labels_threshold = hdbscan.get_labels(threshold=0.5)

# Accès aux composants internes
mst_edges = hdbscan.mst_edges
core_distances = hdbscan.core_dist
```

## Visualisation

Le package inclut des méthodes de visualisation complètes :

```python
import matplotlib.pyplot as plt

# Création d'une figure avec sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# 1. Arbre couvrant minimal (MST)
hdbscan.mst.plot(ax=axes[0, 0], node_colors=labels)

# 2. Hiérarchie avec seuil automatique
hdbscan.linkage_tree.plot(ax=axes[0, 1], percentile=hdbscan.auto_percentile_)

# 3. Arbre condensé
hdbscan.condensed_tree.plot(ax=axes[1, 0])

# 4. Résultats du clustering
ax = axes[1, 1]
uniq = np.unique(labels)
colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(uniq[uniq >= 0]))))

for l in uniq:
    mask = labels == l
    if l == -1:
        ax.scatter(X[mask, 0], X[mask, 1], c='k', marker='x', s=60, label='Noise', alpha=0.8)
    else:
        c = colors[l % len(colors)]
        ax.scatter(X[mask, 0], X[mask, 1], c=[c], s=80, edgecolors='k', label=f'Cluster {l}')

ax.set_title('Clusters finaux')
ax.legend()

plt.tight_layout()
plt.show()
```

## Paramètres

### HDBSCAN

- `min_pts` (int, default=5): Nombre minimum de voisins pour le calcul des core distances
- `min_cluster_size` (int, default=5): Taille minimale d'un cluster

### Méthodes

- `fit(X)`: Apprentissage du modèle sur les données X
- `get_labels(auto=False, percentile=None, threshold=None)`: Obtention des labels de cluster

## Fonctionnalités

- Détection de clusters de formes arbitraires
- Gestion automatique du bruit
- Sélection automatique du seuil de clustering
- Visualisations complètes (MST, hiérarchie, arbre condensé)
- Implémentation modulaire et extensible
- Algorithmes optimisés (Union-Find, Prim pour MST)

## Classes principales

### `HDBSCAN`

Classe principale implémentant l'algorithme complet.

### `MST`

Représente l'arbre couvrant minimal avec méthodes de visualisation.

### `UNION_FIND`

Structure de données pour la gestion des composantes connexes.

### `LINKAGE_TREE`

Gère l'arbre de liaison hiérarchique.

### `CONDENSED_TREE`

Implémente l'arbre condensé pour la sélection de clusters.

## Exécution de la démo

```python
# Exécuter la démonstration complète
from hdbscan_package.demo import main
main()
```

## Notes

- L'algorithme utilise une implémentation de Prim pour la construction du MST
- La sélection automatique du seuil utilise la méthode du plus grand écart
- Les points étiquetés `-1` sont considérés comme du bruit
- Le package est particulièrement efficace pour les données avec bruit et clusters de densité variable

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :

- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Optimiser les performances

## Licence

Ce projet est fourni à des fins éducatives et de recherche.
