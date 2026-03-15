import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


class MST:
    """
    Classe pour visualiser l'Arbre Couvrant Minimum (Minimum Spanning Tree).

    Le MST est une représentation graphique des connexions entre les points
    basées sur les distances de mutual reachability. Chaque arête est colorée
    selon son poids, permettant d'identifier visuellement les structures de densité.
    """

    def __init__(self, edges, X):
        """
        Initialise le visualiseur MST.

        Parameters:
        -----------
        edges : list of tuples (i, j, weight)
            Liste des arêtes du MST avec leurs poids.
        X : array-like de shape (n_samples, 2)
            Coordonnées des points dans l'espace 2D.
        """
        self.edges = edges  # Conserve les arêtes du MST
        self.X = X  # Conserve les données originales

        # Extrait tous les poids pour le scaling des couleurs
        self.weights = [w for _, _, w in edges]

        # Calcule les valeurs min/max pour la normalisation des couleurs
        self.min_w = min(self.weights) if self.weights else 0
        self.max_w = max(self.weights) if self.weights else 0

    def plot(self, ax=None, node_colors=None):
        """
        Trace le MST avec les points et les arêtes colorées par poids.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes sur lesquels tracer. Si None, crée une nouvelle figure.
        node_colors : array-like, optional
            Couleurs pour chaque point. Si None, tous les points sont gris.

        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes contenant le tracé du MST.
        """
        # Créer une nouvelle figure si aucun axe n'est fourni
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        # Afficher les points de données
        if node_colors is not None:
            # Si des couleurs sont fournies, les utiliser avec une colormap
            scatter = ax.scatter(
                self.X[:, 0], self.X[:, 1], c=node_colors, s=50,
                edgecolor='k', alpha=0.3, cmap='tab10')
        else:
            # Sinon, afficher tous les points en gris clair
            ax.scatter(self.X[:, 0], self.X[:, 1],
                       c='lightgray', s=50, alpha=0.3, edgecolor='k')

        # Préparer les segments pour LineCollection (pour tracer toutes les arêtes efficacement)
        lines = []
        line_colors = []
        for u, v, w in self.edges:
            # Crée une ligne entre les points u et v
            lines.append([(self.X[u, 0], self.X[u, 1]),
                         (self.X[v, 0], self.X[v, 1])])

            # Normalise le poids entre 0 et 1 pour la colormap
            norm_w = (w - self.min_w) / (self.max_w -
                                         self.min_w) if self.max_w > self.min_w else 0

            # Convertit la valeur normalisée en couleur via la colormap plasma
            line_colors.append(plt.cm.plasma(norm_w))

        # Tracer toutes les arêtes en une seule fois (plus efficace que plt.plot individuel)
        lc = LineCollection(lines, colors=line_colors,
                            alpha=0.6, linewidths=2)
        ax.add_collection(lc)

        # Ajouter une barre de couleur pour légender les poids
        sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(
            vmin=self.min_w, vmax=self.max_w))
        sm.set_array([])  # Nécessaire pour ScalarMappable
        plt.colorbar(sm, ax=ax, label='Mutual Reachability Distance')

        # Configuration du graphique
        ax.set_title('1. Minimum Spanning Tree (MST)', fontsize=12)
        ax.set_aspect('equal')  # Assure des proportions égales sur les axes
        # Grille en pointillés semi-transparente
        ax.grid(True, linestyle='--', alpha=0.3)

        # Masque les ticks des axes pour un graphique plus propre
        ax.set_xticks([])
        ax.set_yticks([])

        return ax
