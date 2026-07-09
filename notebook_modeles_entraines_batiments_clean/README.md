# Notebook modèles entraînés bâtiments

Cette archive contient une seule version du notebook à ouvrir :

```text
notebooks/notebook_modeles_entraines_batiments.ipynb
```

Ce notebook est déjà exécuté et peut être projeté directement. Il analyse les modèles entraînés de l'application finale : Perceptron, MLP/PMC, RBF et SVM.

## Contenu

```text
notebooks/   notebook unique à ouvrir
graphs/      graphes exportés en PNG
results/     tableaux exportés en CSV
app_final/   application finale avec les modèles entraînés
```

## Installation minimale

```bash
pip install -r requirements_notebook.txt
```

## Lancement

```bash
jupyter notebook notebooks/notebook_modeles_entraines_batiments.ipynb
```

ou avec VS Code : ouvrir directement le fichier `.ipynb`.

## Fichiers importants utilisés par le notebook

```text
app_final/projet_rendu_2_ml/models/models_manifest.json
app_final/projet_rendu_2_ml/models/reports/
```

Les graphes sont déjà disponibles séparément dans `graphs/`, et les résultats tabulaires dans `results/`.


## Ajout pour les questions techniques

Le notebook contient une section dédiée aux questions probables du professeur :
RBF naïf ou K-means, Lloyd, nombre de centres, sigma, type de SVM, logique
un-contre-tous, paramètres des modèles.

Ces informations sont aussi disponibles séparément dans :

```text
docs/questions_techniques_algorithmes.md
results/details_techniques_algorithmes.csv
```
