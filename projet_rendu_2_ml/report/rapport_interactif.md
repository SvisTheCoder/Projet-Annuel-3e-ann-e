# Première ébauche du rapport interactif — rendu 2

## 1. Objectif

Le projet compare quatre modèles de classification :

- modèle linéaire / perceptron de Rosenblatt ;
- perceptron multicouche ;
- Radial Basis Function Network ;
- SVM linéaire.

Les modèles sont d'abord vérifiés sur des cas de test simples, puis ils peuvent
être entraînés sur le dataset d'images de bâtiments exporté en CSV.

## 2. Architecture technique

L'application est divisée en deux parties :

```text
Application, CSV et réseau en C
            ↓
       API compatible C
            ↓
       Modèles en C++
```

Le fichier `ml_api.h` masque les détails C++ derrière un type opaque
`MLModel*`. L'application C appelle des fonctions simples pour créer,
entraîner, évaluer, sauvegarder, charger et utiliser les modèles.

## 3. Préparation des données

Les images sont :

1. converties en niveaux de gris ;
2. redimensionnées en 32 × 32 ;
3. normalisées entre 0 et 1 ;
4. aplaties en 1024 caractéristiques ;
5. enregistrées dans un CSV avec le label en dernière colonne.

Le script `scripts/images_to_csv.py` automatise cette transformation.

## 4. Modèles

### 4.1 Perceptron de Rosenblatt

Le perceptron utilise une stratégie un-contre-tous. Pour la classe étudiée, la
réponse attendue vaut `+1`; pour les autres classes, elle vaut `-1`.

Le score d'une classe est :

```text
score = somme(pixel × poids) + biais
```

En cas d'erreur :

```text
poids = poids + learning_rate × réponse_attendue × exemple
biais = biais + learning_rate × réponse_attendue
```

### 4.2 Perceptron multicouche

Le PMC contient :

- une couche d'entrée ;
- une couche cachée avec sigmoïde ;
- une couche de sortie avec softmax.

L'entraînement est réalisé exemple par exemple par SGD et rétropropagation.

### 4.3 RBF

Le RBF choisit plusieurs exemples comme centres. Une activation gaussienne
mesure la proximité entre une donnée et chaque centre :

```text
proche du centre → activation forte
loin du centre   → activation faible
```

Les activations alimentent ensuite une couche de sortie entraînée par SGD.

### 4.4 SVM

Le SVM est linéaire et entraîné en un-contre-tous. Il utilise une perte hinge
et une régularisation L2. Il cherche à classer les exemples du bon côté de la
frontière avec une marge suffisante.

## 5. Cas de test vérifiés

### Données linéairement séparables

| Modèle | Accuracy |
|---|---:|
| Perceptron | 1.000 |
| PMC | 1.000 |
| RBF | 1.000 |
| SVM | 1.000 |

### XOR

| Modèle | Accuracy |
|---|---:|
| Perceptron | 0.500 |
| PMC | 1.000 |
| RBF | 1.000 |
| SVM | 0.500 |

Ces résultats illustrent la différence entre les modèles linéaires et les
modèles non linéaires. Le perceptron et le SVM linéaire ne peuvent pas séparer
XOR, alors que le PMC et le RBF le peuvent.

## 6. Évaluation sur le dataset constitué

Le dataset est mélangé puis séparé en :

```text
80 % entraînement
20 % test
```

Le programme affiche l'accuracy sur les deux ensembles.

| Modèle | Train accuracy | Test accuracy | Commentaire |
|---|---:|---:|---|
| Perceptron | À mesurer | À mesurer | Baseline linéaire |
| PMC | À mesurer | À mesurer | Modèle non linéaire |
| RBF | À mesurer | À mesurer | Basé sur les distances |
| SVM | À mesurer | À mesurer | Séparation à marge maximale |

Les valeurs seront complétées après ajout du dataset final de bâtiments.

## 7. Sauvegarde et chargement

Chaque modèle peut être écrit dans un fichier texte. Le fichier contient :

- le type du modèle ;
- le nombre de caractéristiques et de classes ;
- les hyperparamètres ;
- les poids, biais, centres ou matrices nécessaires.

Les tests automatiques entraînent chaque modèle, le sauvegardent, le
rechargent et vérifient que l'accuracy reste identique.

## 8. Client / serveur

Le serveur :

1. charge un modèle déjà entraîné ;
2. écoute sur un port TCP ;
3. reçoit un vecteur de caractéristiques ;
4. effectue la prédiction ;
5. renvoie l'indice de la classe.

Le client saisit les caractéristiques, les envoie au serveur et affiche la
réponse. Une prédiction client/serveur a été testée avec le modèle linéaire de
démonstration.

## 9. Commandes de démonstration

```bash
./build/demo --tests
./build/demo --compare data/linear.csv
./build/server models/perceptron_linear.txt 5050
./build/client 127.0.0.1 5050 2
```

## 10. Limites actuelles

- le SVM est uniquement linéaire ;
- les centres RBF ne sont pas calculés par K-means ;
- le split train/test n'est pas stratifié ;
- la résolution 32 × 32 retire une partie de l'information visuelle ;
- le serveur traite un seul client à la fois ;
- le code privilégie la compréhension plutôt que l'optimisation.

## 11. Suite prévue

- intégrer le dataset final ;
- mesurer les résultats des quatre modèles ;
- ajouter des graphiques de comparaison dans le rapport ;
- commenter les écarts entre train et test ;
- tester plusieurs hyperparamètres sans complexifier l'implémentation.
