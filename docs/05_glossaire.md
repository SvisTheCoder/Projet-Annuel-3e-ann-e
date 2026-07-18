# 5 — Glossaire essentiel

| Terme | Définition simple dans ce projet |
|---|---|
| Feature / caractéristique | Une information numérique fournie au modèle. Ici, un pixel normalisé. |
| Label | La bonne réponse connue pour une image du dataset : 0, 1 ou 2. |
| Dataset | L'ensemble d'images étiquetées utilisé pour apprendre et évaluer. |
| CSV canonique | Le fichier tabulaire reproductible obtenu à partir des images sources. |
| Prétraitement | La transformation identique appliquée à chaque image avant le ML. |
| Normalisation | Passage des pixels de l'échelle 0–255 à l'échelle 0–1. |
| Entraînement / fit | Ajustement des paramètres d'un modèle à partir des exemples et labels. |
| Prédiction / inference | Utilisation d'un modèle déjà entraîné sur une nouvelle entrée. |
| Modèle sauvegardé | Fichier qui conserve les paramètres appris afin de prédire plus tard. |
| Seed | Valeur initiale qui rend certaines opérations aléatoires reproductibles. Ici : 42. |
| Split train/test | Séparation des données : une partie apprend, l'autre évalue. |
| Validation | Sous-ensemble utilisé pour comparer des paramètres sans toucher au test final. |
| Accuracy | Nombre de prédictions correctes divisé par le nombre total de prédictions. |
| Matrice de confusion | Tableau qui montre, pour chaque vraie classe, les classes réellement prédites. |
| One-vs-rest | Une stratégie multi-classe où chaque classe est comparée aux autres réunies. |
| Marge | Score relatif à une frontière de décision ; ce n'est pas forcément une probabilité. |
| Perceptron | Classifieur linéaire historique, simple à expliquer. |
| SVM | Classifieur qui cherche une séparation avec une marge entre les classes. |
| MLP | Réseau de neurones avec une couche cachée, capable d'apprendre des relations non linéaires. |
| RBF | Modèle qui évalue la proximité avec des centres au moyen de fonctions gaussiennes. |
| Centre RBF | Exemple ou point représentatif auquel une image est comparée. |
| Sigma RBF | Largeur de la fonction gaussienne : trop petit ou trop grand peut rendre le modèle peu informatif. |
| Eigen | Bibliothèque C++ de calcul matriciel utilisée par `ml_library`. |
| CMake | Outil qui configure les cibles de compilation. |
| MinGW | Chaîne de compilation C/C++ utilisée par CLion sous Windows. |
| Flask | Framework Python qui fournit la page web locale. |
| Pillow | Bibliothèque Python pour ouvrir et transformer les images. |
| `ml_api` | Adaptateur entre les programmes en C et les algorithmes en C++. |
| `predict_cli` | Exécutable C++ qui charge un modèle et renvoie une prédiction JSON. |
| `train_cli` | Exécutable C++ qui entraîne, évalue, sauvegarde et documente un modèle. |
| Manifest | Catalogue JSON qui indique à Flask quels modèles compatibles proposer. |
| stdout / stderr | Sorties séparées d'un programme : ici JSON sur stdout et erreurs sur stderr. |
| Temporaire | Fichier créé seulement pendant une requête web puis supprimé. |
