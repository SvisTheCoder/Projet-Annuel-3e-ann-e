# Guide de compréhension du projet

Ce dossier est fait pour comprendre le projet avant de lire le code. Son but est
de permettre d'expliquer clairement le travail à un jury : le besoin auquel le
projet répond, le chemin suivi par les données, le rôle de chaque application et
les limites réelles des résultats.

## Ordre de lecture conseillé

1. [Vue d'ensemble](01_vue_ensemble.md) : la carte complète du projet.
2. [Données, entraînement et modèles](02_donnees_entrainement_modeles.md) : ce qui se passe avant l'utilisation de l'interface.
3. [Prédire et utiliser les applications](03_prediction_et_applications.md) : ce qui se passe lorsqu'une image est envoyée.
4. [Préparer la soutenance](04_preparer_la_soutenance.md) : un fil conducteur, des réponses courtes et les limites à assumer.
5. [Glossaire](05_glossaire.md) : les mots à maîtriser sans jargon inutile.

## La phrase à retenir

> Nous entraînons hors ligne quatre modèles C++ sur des images de bâtiments
> transformées en 1 024 nombres. Nous sauvegardons les modèles validés, puis
> l'interface web locale charge un de ces modèles existants pour classer une
> nouvelle image parmi Art déco, Art nouveau et Gothique.

## Ce que ce projet est — et n'est pas

| C'est | Ce n'est pas |
|---|---|
| Un classifieur local d'images de bâtiments en trois classes. | Un service cloud ou une application de production à grande échelle. |
| Une implémentation C/C++ des algorithmes, avec Eigen pour les calculs matriciels. | Une application qui utilise TensorFlow, PyTorch ou scikit-learn. |
| Un entraînement reproductible séparé de l'interface. | Un site web qui réentraîne le modèle au moment d'un upload. |
| Une interface Flask qui prépare l'image puis appelle l'exécutable C++. | Une prédiction machine learning faite en Python. |

## Arborescence à connaître

```text
Projet-Annuel-3e-ann-e/
├── ml_library/                 # Les quatre algorithmes C++ réels
├── notebook_modeles_entraines_batiments_clean/
│   └── app_final/projet_rendu_2_ml/  # Application, données et interface web
│       ├── data/                     # CSV canonique
│       ├── models/                   # Modèles, manifest et rapports
│       ├── scripts/                  # Conversion et entraînement
│       ├── src/                      # Exécutables C/C++
│       ├── web/                      # Interface Flask locale
│       └── CMakeLists.txt            # Assemblage des cibles avec CMake
└── docs/                       # Le présent guide pédagogique
```

## Règle de lecture importante

Les flèches des schémas représentent le **sens des données** ou le **sens d'un
appel**. Lorsqu'un fichier est marqué « sauvegardé », il est produit pendant
l'entraînement et seulement lu lors de la prédiction.
