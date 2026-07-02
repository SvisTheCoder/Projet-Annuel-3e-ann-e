# Projet annuel — rendu 2 — modèles ML en C/C++

Ce projet fournit une **application principale en C** qui utilise une
**bibliothèque de modèles écrite en C++** grâce à une petite interface
compatible C (`include/ml_api.h`).

Le code est volontairement simple, détaillé et peu optimisé afin de pouvoir
être relu et réécrit à la main.

## Fonctionnalités livrées

- modèle linéaire / perceptron de Rosenblatt en un-contre-tous ;
- perceptron multicouche avec une couche cachée, softmax et SGD ;
- réseau RBF avec activations gaussiennes ;
- SVM linéaire en un-contre-tous ;
- cas de test linéaire et XOR ;
- chargement de datasets CSV ;
- séparation aléatoire 80 % entraînement / 20 % test ;
- calcul des accuracies train et test ;
- comparaison des quatre modèles sur un même dataset ;
- sauvegarde et chargement des modèles dans un fichier texte ;
- prédiction sur une nouvelle donnée ;
- serveur TCP de prédiction et client TCP ;
- conversion optionnelle de dossiers d’images vers un CSV ;
- première ébauche du rapport dans `report/rapport_interactif.md`.

## Pourquoi l’application est en C mais les modèles en C++ ?

Le menu, le chargement CSV, les tests et le réseau sont en C. Les modèles sont
en C++ car la librairie de départ était en C++.

Le fichier `include/ml_api.h` expose uniquement des fonctions compatibles C :

```c
MLModel* ml_create(...);
int ml_train(...);
int ml_predict(...);
int ml_save(...);
MLModel* ml_load(...);
void ml_free(...);
```

Le code C ne voit donc jamais les objets et les `std::vector` C++.

## Format du dataset CSV

Une ligne contient les caractéristiques, puis le label entier dans la dernière
colonne :

```text
0.12,0.44,0.80,0.31,2
```

Les labels doivent commencer à 0 et être continus :

```text
0, 1, 2, ...
```

## Compilation

### Linux, WSL ou macOS

```bash
cmake -S . -B build
cmake --build build
```

Ou :

```bash
./build_linux.sh
```

### Windows avec CMake et Visual Studio

```powershell
cmake -S . -B build
cmake --build build --config Release
```

Ou double-cliquer sur :

```text
build_windows.bat
```

## Utilisation de la démonstration

### Menu interactif

Linux / WSL :

```bash
./build/demo
```

Windows avec Visual Studio :

```powershell
.\build\Release\demo.exe
```

Le menu permet de :

1. lancer les cas de test ;
2. charger un CSV ;
3. entraîner chacun des quatre modèles ;
4. comparer les modèles ;
5. évaluer le modèle courant ;
6. sauvegarder ou charger un modèle ;
7. prédire une nouvelle donnée.

### Tests automatiques

```bash
./build/demo --tests
```

Résultat attendu :

- les quatre modèles réussissent le cas linéaire ;
- le PMC et le RBF réussissent XOR ;
- le perceptron et le SVM linéaires restent limités sur XOR ;
- chaque modèle est sauvegardé puis rechargé avec le même résultat.

### Comparer les modèles sur un CSV

```bash
./build/demo --compare data/linear.csv
```

## Client / serveur

Un modèle d’exemple est fourni :

```text
models/perceptron_linear.txt
```

Démarrer le serveur :

```bash
./build/server models/perceptron_linear.txt 5050
```

Dans un autre terminal, envoyer une donnée à deux caractéristiques :

```bash
./build/client 127.0.0.1 5050 2
```

Le client demande les deux valeurs puis affiche la classe prédite.

Le protocole est volontairement minimal :

1. nombre de caractéristiques ;
2. tableau de valeurs `double` ;
3. classe prédite renvoyée par le serveur.

Le serveur traite un client à la fois.

## Dataset d’images

La bibliothèque reçoit des nombres et ne lit pas directement les fichiers
PNG/JPEG. Le script Python optionnel transforme un dossier organisé par
classes :

```text
dataset/
  art_deco/
  art_nouveau/
  gothique/
```

Commande :

```bash
python scripts/images_to_csv.py dataset data/batiments.csv 32
```

Chaque image est :

1. convertie en niveaux de gris ;
2. redimensionnée en 32 × 32 ;
3. normalisée entre 0 et 1 ;
4. aplatie en 1024 valeurs ;
5. enregistrée dans le CSV avec son label.

## Organisation du projet

```text
include/
  ml_api.h        API C vers les modèles C++
  dataset.h       structures et fonctions CSV
  net_compat.h    compatibilité réseau Windows/Linux
  tests.h         lancement des cas de test

src/
  ml_api.cpp      quatre modèles et sauvegarde/chargement
  dataset.c       lecture, écriture et séparation du dataset
  tests.c         cas linéaire, XOR et test de persistance
  demo.c          interface console en C
  server.c        serveur de prédiction
  client.c        client de prédiction

data/             petits CSV de démonstration
models/           modèles sauvegardés
scripts/          conversion optionnelle des images
report/           première ébauche du rapport
```

## Limites assumées

- le SVM est linéaire ;
- les centres du RBF sont choisis parmi les exemples, sans K-means ;
- le redimensionnement des images peut perdre beaucoup d’information ;
- le split train/test n’est pas stratifié ;
- le serveur est séquentiel ;
- les données réseau en `double` supposent client et serveur sur une
  architecture compatible ;
- le projet privilégie la compréhension plutôt que la performance.
