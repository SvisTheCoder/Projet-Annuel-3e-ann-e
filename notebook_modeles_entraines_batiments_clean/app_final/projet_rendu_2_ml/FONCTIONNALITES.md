# Fonctionnalités présentes dans le rendu

## Modèles

- [x] Perceptron de Rosenblatt multi-classe par un-contre-tous
- [x] PMC binaire avec une couche cachée, sigmoïde et rétropropagation
- [x] Adaptation multi-classe des modèles par un-contre-tous dans `ml_api.cpp`
- [x] RBF avec centres simples et fonctions gaussiennes
- [x] SVM linéaire avec perte hinge

## Données

- [x] Lecture CSV
- [x] Labels entiers en dernière colonne
- [x] Split 80/20 reproductible
- [x] Accuracy train/test
- [x] Conversion optionnelle images vers CSV

## Démonstration

- [x] Cas linéaire
- [x] Cas XOR
- [x] Comparaison automatique des quatre modèles
- [x] Menu console en C

## Persistance

- [x] Sauvegarde texte de chaque type de modèle
- [x] Chargement de chaque type de modèle
- [x] Vérification automatique après rechargement

## Utilisation des modèles

- [x] Entraînement reproductible avec `train_cli`
- [x] Prédiction non interactive avec `predict_cli`
- [x] Interface web locale Flask
- [x] Manifest des modèles disponibles

## Livrables

- [x] Sources complètes
- [x] CMakeLists.txt
- [x] Scripts de compilation Windows/Linux
- [x] README
- [x] Guide de lecture du code
- [x] Première ébauche du rapport
