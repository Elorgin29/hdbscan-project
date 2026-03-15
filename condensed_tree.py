import numpy as np
import matplotlib.pyplot as plt
from .union_find import UNION_FIND


class CONDENSED_TREE:
    """
    Classe représentant l'arbre condensé de HDBSCAN.

    L'arbre condensé est une structure hiérarchique qui capture la naissance et la mort
    des clusters à différents niveaux de seuil de distance. Il permet d'évaluer la stabilité
    des clusters et de sélectionner automatiquement les clusters les plus pertinents.
    """

    def __init__(self, mst_edges, min_cluster_size, X):
        """
        Initialise l'arbre condensé HDBSCAN.

        Parameters:
        -----------
        mst_edges : list of tuples (i, j, weight)
            Liste des arêtes de l'arbre couvrant minimum (MST) avec leurs poids.
        min_cluster_size : int
            Taille minimale pour qu'un groupe soit considéré comme un cluster valide.
        X : array-like de shape (n_samples, n_features)
            Les données d'origine utilisées pour le clustering.
        """
        self.mst_edges = mst_edges  # Conserve les arêtes du MST
        self.min_cluster_size = min_cluster_size  # Seuil de taille minimale
        self.X = X  # Données originales
        # Extraction de tous les poids pour analyse statistique
        self.weights = [w for _, _, w in mst_edges]

    def _get_num_clusters(self, thresh):
        """
        Calcule le nombre de clusters valides pour un seuil de distance donné.

        Cette méthode simule la coupe du MST à un niveau de distance spécifique
        et compte combien de clusters de taille suffisante émergent.

        Parameters:
        -----------
        thresh : float
            Seuil de distance : seules les arêtes avec poids <= thresh sont conservées.

        Returns:
        --------
        num_clusters : int
            Nombre de clusters ayant au moins min_cluster_size points.
        """
        # Initialisation de la structure UNION-FIND pour chaque point
        uf = UNION_FIND(len(self.X))

        # Trie les arêtes du MST par poids croissant
        sorted_mst = sorted(self.mst_edges, key=lambda x: x[2])

        # Fusionne les points connectés par des arêtes de poids <= seuil
        for u, v, w in sorted_mst:
            if w <= thresh:
                uf.union(u, v)

        # Récupère les labels des composantes connectées
        labels = np.array(uf.get_labels())

        # Compte la taille de chaque cluster
        uniq, counts = np.unique(labels, return_counts=True)

        # Filtre les clusters trop petits et retourne le nombre restant
        valid = uniq[counts >= self.min_cluster_size]
        return len(valid)

    def plot(self, ax=None, percentiles=[10, 30, 50, 70, 90]):
        """
        Trace la courbe du nombre de clusters en fonction du seuil de distance.

        Cette visualisation montre comment le paysage des clusters évolue
        lorsque l'on modifie le seuil de coupure du MST. Elle aide à identifier
        les seuils stables où le nombre de clusters reste relativement constant.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes sur lesquels tracer la courbe. Si None, crée une nouvelle figure.
        percentiles : list of float, default=[10, 30, 50, 70, 90]
            Liste des percentiles des poids du MST à évaluer.

        Returns:
        --------
        ax : matplotlib.axes.Axes
            Axes contenant le tracé.
        """
        # Crée une nouvelle figure si aucun axe n'est fourni
        if ax is None:
            _, ax = plt.subplots()

        # Convertit les percentiles en valeurs de seuil réelles
        threshs = np.percentile(self.weights, percentiles)

        # Calcule le nombre de clusters pour chaque seuil
        counts = []
        for thr in threshs:
            counts.append(self._get_num_clusters(thr))

        # Trace la courbe nombre de clusters vs seuil de distance
        # '-s' = ligne avec des carrés aux points
        ax.plot(threshs, counts, '-s')

        # Inverse l'axe X car des seuils plus hauts donnent moins de clusters
        ax.invert_xaxis()

        # Configuration du graphique
        ax.set_title('3. Condensed Tree (Auto Threshold)', fontsize=12)
        ax.set_xlabel('Distance Threshold')
        ax.set_ylabel('Number of Clusters')

        return ax
