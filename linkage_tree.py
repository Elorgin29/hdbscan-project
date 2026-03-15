import numpy as np
import matplotlib.pyplot as plt


class LINKAGE_TREE:
    """
    Classe pour visualiser l'arbre de liaison hiérarchique d'HDBSCAN.

    L'arbre de liaison montre l'évolution des distances de fusion lors de la construction
    progressive du clustering hiérarchique. Chaque point représente une fusion entre
    deux clusters, et la courbe permet d'identifier les seuils naturels de coupure.
    """

    def __init__(self, sorted_edges):
        """
        Initialise l'arbre de liaison à partir des arêtes triées du MST.

        Parameters:
        -----------
        sorted_edges : list of tuples (i, j, weight)
            Liste des arêtes du MST triées par poids croissant.
            Chaque arête représente une fusion dans la hiérarchie.
        """
        # Extrait uniquement les poids des arêtes pour l'analyse de la hiérarchie
        self.sorted_weights = [w for _, _, w in sorted_edges]

    def plot(self, ax=None, percentile=None):
        """
        Trace la courbe de l'arbre de liaison (distances de fusion vs étapes).

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes sur lesquels tracer. Si None, crée une nouvelle figure.
        percentile : float, optional
            Percentile à afficher comme ligne horizontale de référence.

        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes contenant le tracé de l'arbre de liaison.
        """
        # Crée une nouvelle figure si aucun axe n'est fourni
        if ax is None:
            _, ax = plt.subplots()

        # Trace la courbe des poids triés en fonction de l'étape de fusion
        # Chaque point correspond à une fusion dans la hiérarchie
        ax.plot(range(len(self.sorted_weights)),
                self.sorted_weights, '-o', markersize=3)

        # Si un percentile est spécifié, trace une ligne horizontale de référence
        if percentile is not None:
            # Calcule la valeur de seuil correspondant au percentile
            thresh = np.percentile(self.sorted_weights, percentile)

            # Ajoute la ligne rouge en pointillés avec légende
            ax.axhline(thresh, color='red', linestyle='--',
                       label=f'{percentile:.1f}ᵉ percentile')
            ax.legend()

        # Configuration du graphique
        ax.set_title('2. Hierarchy (Auto Threshold)', fontsize=12)
        ax.set_xlabel('Merge Step')  # Numéro de l'étape de fusion
        # Distance de mutual reachability à la fusion
        ax.set_ylabel('Distance')

        return ax
