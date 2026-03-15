import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets as data
from hdbscan_package import HDBSCAN


def main():
    """
    Fonction principale de démonstration de l'algorithme HDBSCAN.

    Cette fonction illustre le workflow complet :
    1. Génération de données synthétiques complexes
    2. Application d'HDBSCAN avec sélection automatique du seuil
    3. Visualisation des étapes intermédiaires et des résultats finaux
    """

    # ========== GÉNÉRATION DE DONNÉES SYNTHÉTIQUES ==========
    # Crée des données composées de deux demi-lunes et deux blobs gaussiens
    # Cela permet de tester la capacité d'HDBSCAN à gérer des formes non-convexes

    # Première structure : deux demi-lunes imbriquées (non linéairement séparables)
    moons, _ = data.make_moons(n_samples=50, noise=0.05, random_state=42)

    # Deuxième structure : deux gaussiennes distinctes (clusters convexes)
    blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)],
                               cluster_std=0.25, random_state=42)

    # Combine les deux ensembles de données pour créer un jeu complexe
    X = np.vstack([moons, blobs])

    # ========== CONFIGURATION ET ENTRAÎNEMENT DU MODÈLE ==========
    # Paramètres clés d'HDBSCAN :
    # - min_pts : contrôle la densité minimale locale (distance au cœur)
    # - min_cluster_size : taille minimale pour considérer un groupe comme cluster valide

    min_pts = 5
    min_cluster_size = 5

    # Instanciation et entraînement du modèle HDBSCAN
    hdbscan = HDBSCAN(min_pts=min_pts, min_cluster_size=min_cluster_size)
    hdbscan.fit(X)

    # ========== CLUSTERING AVEC SÉLECTION AUTOMATIQUE DU SEUIL ==========
    # Utilise la sélection automatique basée sur le gap dans la distribution des poids
    # Cette méthode identifie naturellement le seuil optimal de coupure

    labels_auto = hdbscan.get_labels(auto=True)

    # Affichage des métriques du clustering
    print("Percentile sélectionné automatiquement :",
          round(hdbscan.auto_percentile_, 2))
    print("Clusters détectés :", len(set(labels_auto[labels_auto >= 0])))
    print("Points de bruit   :", np.sum(labels_auto == -1))

    # ========== VISUALISATION DES RÉSULTATS ==========
    # Crée une grille 2x2 de sous-graphiques pour montrer :
    # 1. Le MST avec les points colorés par cluster final
    # 2. L'arbre de liaison avec le seuil auto
    # 3. L'arbre condensé
    # 4. Les clusters finaux

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Visualisation du MST (Arbre Couvrant Minimum)
    # Les nœuds sont colorés selon leur cluster final pour voir la structure
    hdbscan.mst.plot(ax=axes[0, 0], node_colors=labels_auto)

    # 2. Visualisation de l'arbre de liaison hiérarchique
    # Affiche la ligne du percentile utilisé pour la coupure automatique
    hdbscan.linkage_tree.plot(
        ax=axes[0, 1], percentile=hdbscan.auto_percentile_)

    # 3. Visualisation de l'arbre condensé
    # Montre le nombre de clusters en fonction du seuil de distance
    hdbscan.condensed_tree.plot(ax=axes[1, 0])

    # 4. Visualisation des clusters finaux
    ax = axes[1, 1]
    uniq = np.unique(labels_auto)

    # Génère une palette de couleurs pour les clusters
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(uniq[uniq >= 0]))))

    # Trace chaque cluster avec sa couleur
    for l in uniq:
        mask = labels_auto == l
        if l == -1:
            # Points de bruit : croix noires
            ax.scatter(X[mask, 0], X[mask, 1], c='k', marker='x',
                       s=60, label='Noise', alpha=0.8)
        else:
            # Points de cluster : couleurs vives avec bordure noire
            c = colors[l % len(colors)]
            ax.scatter(X[mask, 0], X[mask, 1], c=[c], s=80,
                       edgecolors='k', label=f'Cluster {l}')

    # Configuration du graphique final
    ax.set_title('4. Clusters (sélection auto)', fontsize=12)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ajuste la mise en page pour éviter les chevauchements
    plt.tight_layout(pad=2.0)
    plt.show()


if __name__ == "__main__":
    # Exécute la fonction principale uniquement si le script est lancé directement
    main()
