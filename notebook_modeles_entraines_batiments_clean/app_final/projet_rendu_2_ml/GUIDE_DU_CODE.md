# Guide de lecture du code

## Ordre conseillé

1. `include/ml_api.h`
2. `src/demo.c`
3. `src/dataset.c`
4. `src/tests.c`
5. `src/ml_api.cpp`
6. `src/train_cli.cpp`
7. `src/predict_cli.cpp`
8. `web/app.py`

## 1. API entre C et C++

`ml_api.h` définit un type opaque :

```c
typedef struct MLModel MLModel;
```

Le programme C sait qu’un modèle existe, mais il ne connaît pas son contenu.
Seul `ml_api.cpp` connaît la vraie structure.

## 2. Modèle linéaire

Dans `train_perceptron`, chaque classe est traitée en un-contre-tous :

```text
vraie classe : +1
autres classes : -1
```

En cas d’erreur :

```text
poids = poids + learning_rate × réponse_attendue × exemple
biais = biais + learning_rate × réponse_attendue
```

## 3. PMC

Le PMC de `ml_library` effectue, pour chaque exemple :

1. propagation vers la couche cachée ;
2. calcul de la sortie avec une sigmoïde ;
3. calcul de l’erreur ;
4. rétropropagation ;
5. mise à jour des poids.

`ml_api.cpp` crée un PMC binaire par classe pour obtenir une classification
multi-classe en un-contre-tous.

## 4. RBF

`train_rbf` choisit quelques exemples comme centres. Pour chaque nouvelle
donnée, l’activation d’un centre dépend de la distance :

```text
proche du centre : activation proche de 1
loin du centre   : activation proche de 0
```

Les activations sont ensuite données à une couche de sortie entraînée par SGD.

## 5. SVM

`train_svm` entraîne un SVM linéaire par classe. Il cherche non seulement à
classer correctement, mais aussi à obtenir une marge d’au moins 1.

## 6. Sauvegarde

`ml_save` écrit tous les paramètres dans un fichier texte. `ml_load` recrée le
modèle et recharge les tableaux. Les cas de test vérifient automatiquement que
les prédictions restent identiques après rechargement.

## 7. Dataset

`dataset_load_csv` fait deux passages :

1. compter les lignes et colonnes ;
2. allouer les tableaux puis convertir les valeurs.

`dataset_split` mélange des indices et recopie 80 % des lignes dans train et
20 % dans test.

## 8. Applications à présenter

- `demo.exe --tests` vérifie les cas simples et la sauvegarde des modèles ;
- `train_cli.exe` entraîne un modèle à partir du CSV ;
- `predict_cli.exe` charge un modèle et prédit une classe ;
- Flask prépare une image et appelle `predict_cli.exe`.

Flask ne contient pas une deuxième implémentation du machine learning.
