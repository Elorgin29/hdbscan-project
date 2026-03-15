import numpy as np
from .distance import distance_matrix
from .union_find import UNION_FIND


class HDBSCAN:
    """
    Implémentation de l'algorithme HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

    HDBSCAN combine la notion de densité avec le clustering hiérarchique pour identifier des clusters de densités variables
    et marquer les points isolés comme du bruit (-1).
    """

    def __init__(self, min_pts=5, min_cluster_size=5):
        """
        Initialise l'instance HDBSCAN.

        Parameters:
        -----------
        min_pts : int, default=5
            Nombre minimal de points utilisé pour calculer la distance au cœur.
            Correspond au paramètre "min_samples" dans la littérature HDBSCAN.
        min_cluster_size : int, default=5
            Taille minimale pour qu'un groupe soit considéré comme un cluster valide.
        """
        self.min_pts = min_pts
        self.min_cluster_size = min_cluster_size
        self.X = None  # Données d'entraînement
        self.core_dist = None  # Distances au cœur pour chaque point
        self.mrd = None  # Matrice des distances de mutual reachability
        self.mst_edges = None  # Arêtes de l'arbre couvrant minimum (MST)
        self.mst = None  # Objet MST (pour visualisation potentielle)
        self.linkage_tree = None  # Arbre de liaison hiérarchique
        self.condensed_tree = None  # Arbre condensé (hérarchie HDBSCAN)
        self.auto_percentile_ = None  # Percentile auto-sélectionné pour le seuil

    def fit(self, X):
        """
        Applique l'algorithme HDBSCAN sur les données X.

        Parameters:
        -----------
        X : array-like de shape (n_samples, n_features)
            Les données d'entraînement.

        Returns:
        --------
        self : object
            Retourne l'instance elle-même.
        """
        self.X = X

        # Étape 1 : Calcul des distances au cœur (core distances)
        self.core_dist = self._core_distances()

        # Étape 2 : Construction de la matrice des distances de mutual reachability
        # Cette transformation garantit la stabilité du clustering hiérarchique
        self.mrd = self._mutual_reachability_distances()

        # Étape 3 : Construction de l'arbre couvrant minimum (MST)
        # Le MST connecte tous les points avec le poids total minimal
        self.mst_edges = self._minimum_spanning_tree()

        # Import des modules spécifiques pour les étapes suivantes
        from .mst import MST
        from .linkage_tree import LINKAGE_TREE
        from .condensed_tree import CONDENSED_TREE

        # Étape 4 : Construction de l'arbre de liaison hiérarchique à partir du MST
        self.mst = MST(self.mst_edges, self.X)
        sorted_edges = sorted(self.mst_edges, key=lambda x: x[2])
        self.linkage_tree = LINKAGE_TREE(sorted_edges)

        # Étape 5 : Construction de l'arbre condensé (caractéristique principale d'HDBSCAN)
        self.condensed_tree = CONDENSED_TREE(
            self.mst_edges, self.min_cluster_size, self.X)

        # Étape 6 : Sélection automatique du percentile pour le seuil de coupure
        self.auto_percentile_ = self._auto_select_percentile_via_gap()
        return self

    def _core_distances(self):
        """
        Calcule la distance au cœur pour chaque point.

        La distance au cœur d'un point est la distance vers son min_pts-ième voisin le plus proche.
        Cette mesure est fondamentale pour estimer la densité locale.

        Returns:
        --------
        core_dist : ndarray de shape (n_samples,)
            Distance au cœur pour chaque point.
        """
        # Calcul de la matrice de distance complète (symétrique)
        D = distance_matrix(self.X)

        # Trie chaque ligne par ordre croissant et prend la valeur au rang min_pts
        # Le premier élément est la distance 0 (vers soi-même), d'où le slicing
        return np.sort(D, axis=1)[:, self.min_pts]

    def _mutual_reachability_distances(self):
        """
        Construit la matrice des distances de mutual reachability.

        La distance de mutual reachability entre deux points i et j est :
        max(core_dist[i], core_dist[j], dist(i,j))

        Cette transformation garantit que les points de faible densité (grande core_dist)
        doivent être traversés pour rejoindre les points de haute densité.

        Returns:
        --------
        mrd : ndarray de shape (n_samples, n_samples)
            Matrice des distances de mutual reachability.
        """
        # Calcul de la matrice de distance euclidienne standard
        D = distance_matrix(self.X)

        # Expansion des vecteurs de core distances pour la vectorisation
        core_i = self.core_dist[:, np.newaxis]  # Shape: (n_samples, 1)
        core_j = self.core_dist[np.newaxis, :]  # Shape: (1, n_samples)

        # Calcul de la distance de mutual reachability : max(core_i, core_j, D)
        return np.maximum(core_i, np.maximum(core_j, D))

    def _minimum_spanning_tree(self):
        """
        Construit l'arbre couvrant minimum (MST) à partir de la matrice MRD.

        Utilise l'algorithme de Prim pour trouver l'ensemble d'arêtes qui connecte
        tous les points avec le poids total minimal, basé sur les distances de mutual reachability.

        Returns:
        --------
        edges : list of tuples (i, j, weight)
            Liste des arêtes du MST triées par poids croissant.
        """
        n = self.mrd.shape[0]  # Nombre de points

        # Structures de données pour l'algorithme de Prim
        key = [float('inf')] * n  # Poids minimal pour connecter chaque sommet
        parent = [-1] * n  # Parent de chaque sommet dans le MST
        visited = [False] * n  # Marque les sommets déjà inclus dans le MST

        # Commence depuis le premier sommet
        key[0] = 0

        # Itération principale de Prim (n-1 fois pour connecter tous les sommets)
        for _ in range(n):
            # Trouver le sommet non visité avec la clé minimale
            min_key = float('inf')
            u = -1
            for i in range(n):
                if not visited[i] and key[i] < min_key:
                    min_key = key[i]
                    u = i

            # Si aucun sommet trouvé, on sort (graphe non connexe)
            if u == -1:
                break

            # Marquer le sommet comme visité
            visited[u] = True

            # Mettre à jour les clés des voisins non visités
            # Pour chaque sommet v non visité, si la distance mrd[u,v] est plus petite
            # que la clé actuelle de v, mettre à jour la clé et définir u comme parent
            for v in range(n):
                if not visited[v] and self.mrd[u, v] < key[v]:
                    key[v] = self.mrd[u, v]
                    parent[v] = u

        # Reconstruction des arêtes à partir du tableau parent
        # parent[i] = j signifie qu'il y a une arête entre i et j
        edges = []
        for i in range(1, n):  # Commencer à 1 car le sommet 0 est la racine
            if parent[i] != -1:
                edges.append((parent[i], i, key[i]))

        # Trier les arêtes par poids croissant pour l'étape suivante
        edges.sort(key=lambda x: x[2])
        return edges

    def _auto_select_percentile_via_gap(self):
        """
        Sélectionne automatiquement le percentile optimal pour le seuil de coupure.

        Cette méthode recherche le plus grand écart (gap) entre les poids consécutifs
        du MST et utilise ce "saut" naturel pour déterminer un seuil robuste.

        Returns:
        --------
        percentile : float
            Le percentile sélectionné (entre 50% et 99.9%).
        """
        # Extrait et trie les poids des arêtes du MST
        weights = sorted([w for _, _, w in self.mst_edges])

        # Cas dégénéré : trop peu d'arêtes
        if len(weights) < 2:
            return 90.0

        # Calcul des écarts entre poids consécutifs
        gaps = np.diff(weights)

        # Trouver l'indice du plus grand écart
        max_gap_idx = np.argmax(gaps)

        # Le seuil est le point milieu entre les deux poids encadrant le gap
        threshold = (weights[max_gap_idx] + weights[max_gap_idx + 1]) / 2.0

        # Convertir le seuil en percentile
        all_weights = [w for _, _, w in self.mst_edges]
        percentile = 100.0 * \
            np.searchsorted(np.sort(all_weights), threshold,
                            side='right') / len(all_weights)

        # Limiter le percentile entre 50% et 99.9% pour éviter les valeurs extrêmes
        return min(99.9, max(50.0, percentile))

    def get_labels(self, percentile=None, threshold=None, auto=False):
        """
        Extrait les labels de clusters à partir de l'arbre condensé.

        Permutateur flexible : peut utiliser un percentile, un seuil direct,
        ou la sélection automatique basée sur le gap.

        Parameters:
        -----------
        percentile : float, optional
            Percentile des poids du MST utilisé comme seuil de coupure.
        threshold : float, optional
            Seuil direct sur les poids du MST.
        auto : bool, default=False
            Si True, utilise le percentile calculé automatiquement.

        Returns:
        --------
        labels : ndarray de shape (n_samples,)
            Labels des clusters. Les points de bruit sont étiquetés -1.
        """
        # Utiliser le mode auto si demandé
        if auto:
            percentile = self.auto_percentile_

        # Si un percentile est fourni, convertir en seuil
        if percentile is not None:
            weights = [w for _, _, w in self.mst_edges]
            threshold = np.percentile(weights, percentile)

        # Vérification qu'un seuil est bien défini
        if threshold is None:
            raise ValueError(
                "Provide either percentile, threshold, or set auto=True")

        # Extraction des clusters basée sur le seuil
        return self._extract_clusters_with_threshold(threshold)

    def _extract_clusters_with_threshold(self, thresh):
        """
        Extrait les clusters en coupant le MST à un seuil donné.

        Cette méthode utilise une structure UNION-FIND pour connecter les points
        dont les arêtes ont un poids <= seuil, puis filtre les clusters trop petits.

        Parameters:
        -----------
        thresh : float
            Seuil maximal de poids pour qu'une arête connecte deux points.

        Returns:
        --------
        labels : ndarray de shape (n_samples,)
            Labels des clusters. Les points de bruit sont étiquetés -1.
        """
        # Initialisation de la structure UNION-FIND pour chaque point
        uf = UNION_FIND(len(self.X))

        # Parcourir toutes les arêtes du MST triées par poids croissant
        sorted_mst = sorted(self.mst_edges, key=lambda x: x[2])
        for u, v, w in sorted_mst:
            # Si le poids est inférieur ou égal au seuil, on fusionne les deux points
            if w <= thresh:
                uf.union(u, v)

        # Récupération des labels initiaux (chaque composante connectée a un label unique)
        labels = np.array(uf.get_labels())

        # Comptage de la taille de chaque cluster
        uniq, counts = np.unique(labels, return_counts=True)

        # Garder uniquement les clusters ayant au moins min_cluster_size points
        valid = uniq[counts >= self.min_cluster_size]

        # Réaffectation des labels : 0, 1, 2... pour les clusters valides, -1 pour le bruit
        out = np.full(len(self.X), -1, dtype=int)
        for idx, l in enumerate(valid):
            out[labels == l] = idx

        return out
