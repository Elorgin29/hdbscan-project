class UNION_FIND:
    """
    Structure de données Union-Find (ou Disjoint Set Union - DSU).

    Cette structure permet de gérer efficacement des partitions d'éléments et
    d'effectuer les opérations de fusion (union) et de recherche de représentant
    (find) avec une complexité quasi-constante grâce à la compression de chemins
    et à l'union par taille.

    Utilisée dans HDBSCAN pour regrouper les points en clusters lors de la
    coupe du MST à un seuil donné.
    """

    def __init__(self, n):
        """
        Initialise la structure Union-Find pour n éléments.

        Parameters:
        -----------
        n : int
            Nombre d'éléments distincts (points).
        """
        self.parent = list(
            range(n))  # Chaque élément est son propre parent au départ
        self.size = [1] * n  # Taille de chaque ensemble (arbre)

    def find(self, u):
        """
        Trouve le représentant (la racine) de l'ensemble contenant u.

        Implémente la compression de chemins : aplatit l'arbre pendant la recherche
        pour accélérer les futures requêtes.

        Parameters:
        -----------
        u : int
            Indice de l'élément à rechercher.

        Returns:
        --------
        root : int
            Le représentant de l'ensemble de u.
        """
        # Si u n'est pas sa propre racine, on le rattache directement à la racine
        if self.parent[u] != u:
            # Récursion avec compression
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        """
        Fusionne les ensembles contenant u et v.

        Utilise l'union par taille : attache l'arbre plus petit sous la racine
        de l'arbre plus grand pour minimiser la profondeur.

        Parameters:
        -----------
        u, v : int
            Indices des éléments à fusionner.
        """
        # Trouver les racines des deux ensembles
        ru = self.find(u)
        rv = self.find(v)

        # Si ce sont déjà le même ensemble, rien à faire
        if ru != rv:
            # Assurer que ru est la racine du plus grand arbre
            if self.size[ru] < self.size[rv]:
                ru, rv = rv, ru  # Échange les racines

            # Attacher rv sous ru et mettre à jour la taille
            self.parent[rv] = ru
            self.size[ru] += self.size[rv]

    def get_labels(self):
        """
        Retourne les labels finaux pour tous les éléments.

        Effectue une compression de chemins sur tous les éléments pour s'assurer
        que chaque élément pointe directement vers sa racine finale.

        Returns:
        --------
        labels : list of int
            Liste des représentants pour chaque élément (0 à n-1).
        """
        # Retourne les labels finaux en compressant les chemins
        return [self.find(i) for i in range(len(self.parent))]
