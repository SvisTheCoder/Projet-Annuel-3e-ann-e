# 1. Objectif du projet

Ce projet réalise une classification locale d’images de bâtiments en trois
classes :

- `0` : Art déco ;
- `1` : Art nouveau ;
- `2` : Gothique.

Les algorithmes sont écrits en C++ dans la bibliothèque sœur `ml_library`.
L’application principale, les tests et l’adaptateur d’API permettent de les
utiliser depuis du C. Une interface web Flask permet aussi de téléverser une ou
plusieurs images et d’afficher les prédictions localement.

Le projet ne dépend pas de Google Cloud Storage. Le dataset, le CSV, les
modèles et les prédictions restent sur la machine locale.

# 2. Architecture

Le chemin d’une prédiction depuis le site est :

```text
Navigateur
→ Flask / Pillow
→ predict_cli.exe
→ ml_api.cpp
→ ml_library
→ Perceptron / MLP / RBF / SVM
→ JSON
→ interface web
```

- **Navigateur** : envoie les images et affiche les résultats.
- **Flask (`web/app.py`)** : lit le manifest, valide l’upload, applique le
  prétraitement Pillow et lance `predict_cli.exe`. Flask n’entraîne jamais de
  modèle.
- **Pillow (`web/image_processing.py`)** : transforme chaque image en 1 024
  valeurs normalisées.
- **`predict_cli.exe`** : charge un modèle sauvegardé, lit exactement 1 024
  valeurs et écrit une réponse JSON sur stdout.
- **`ml_api.h`** : interface compatible C. Elle expose la création,
  l’entraînement, la prédiction, l’évaluation, la sauvegarde et le chargement
  sans exposer les classes C++.
- **`ml_api.cpp`** : adaptateur entre l’API C et les vraies classes de
  `ml_library`. La classification multiclasse utilise un modèle binaire par
  classe, selon le principe one-vs-rest.
- **`ml_library`** : contient les implémentations Perceptron, MLP, RBF et SVM.
- **`models/models_manifest.json`** : catalogue des modèles proposés par le
  site. Il contient leurs noms, fichiers, dimensions, métriques et types de
  score.

Le chemin d’entraînement officiel est séparé du site :

```text
CSV → train_cli.exe → ml_api.cpp → ml_library → modèle + rapport JSON
```

# 3. Dataset local

Le dataset source actuel est :

```text
data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon
```

Il contient 1 852 images :

| Classe | Label | Images |
|---|---:|---:|
| Art déco | 0 | 554 |
| Art nouveau | 1 | 745 |
| Gothique | 2 | 553 |

Le même prétraitement Python est utilisé pour générer le CSV et pour traiter
les images envoyées à Flask :

1. correction de l’orientation EXIF avec `ImageOps.exif_transpose` ;
2. conversion en niveaux de gris ;
3. redimensionnement en 32 × 32 ;
4. interpolation `Image.Resampling.LANCZOS` ;
5. conversion en nombres flottants et normalisation dans `[0, 1]` ;
6. aplatissement ligne par ligne en exactement 1 024 valeurs.

Le CSV canonique est :

```text
data\batiments_3_classes.csv
```

Chaque ligne contient 1 024 caractéristiques, suivies du label. Le dataset
source et le CSV sont des fichiers lourds : ils doivent rester locaux ou être
gérés avec Git LFS si l’équipe décide de les partager. Aucun script ne doit les
pousser automatiquement.

# 4. Cycle de vie des modèles

| Outil | Quand l’utiliser | Ce qu’il fait | Utilisé par le site ? |
|---|---|---|---|
| `demo.exe` | test manuel ou expérimentation | charge un CSV, entraîne, évalue, sauvegarde et recharge | non |
| `train_cli.exe` | entraînement reproductible d’un modèle | crée un modèle et un rapport JSON, puis valide le rechargement | non |
| `train_all_models.py` | régénération des quatre références | orchestre `train_cli.exe`, crée les rapports et remplace le manifest | non |
| `tune_cli.exe` | comparaison limitée de paramètres | compare des candidats sur une validation interne et écrit un rapport ; il ne sauvegarde pas de modèle final | non |
| `predict_cli.exe` | prédiction C++ sur une image déjà transformée en valeurs | charge le modèle et retourne uniquement du JSON sur stdout | oui |
| Flask | interface web locale | upload, prétraitement, appel de `predict_cli` et affichage | oui |

`train_all_models.py` ne contient aucun algorithme de machine learning Python.
Il appelle le véritable exécutable C++ avec `subprocess.run(..., shell=False)`,
lit ses rapports et construit le manifest. Il est utile à tout développeur qui
doit maintenir ou reproduire les modèles ; ce n’est pas un outil réservé à
Codex.

Le script doit être lancé lors d’un changement de dataset, d’un changement des
paramètres officiels, de la création d’une version officielle ou d’une
régénération reproductible. Il ne doit pas être lancé à chaque démarrage du
site ni pour chaque prédiction.

## Règle vérifiée dans le code actuel

Les règles « le site n’entraîne jamais » et « Flask lit le manifest puis appelle
`predict_cli` » sont exactes. La régénération officielle reproduit Perceptron
v1, MLP v2, RBF v1 et SVM v1. Pour MLP v2, elle transmet explicitement 12
epochs, un learning rate de 0,02 et 32 neurones cachés à `train_cli.exe`.

Flask exige toujours `feature_count=1024` et `class_count=3` dans le manifest.
Il transmet également ce nombre de classes à `predict_cli` avec
`--expected-class-count 3`. Le CLI compare alors la valeur déclarée au nombre
réel de classes lu dans le fichier modèle.

# 5. Modèles disponibles

Le manifest courant propose les modèles suivants :

| Modèle | Nom affiché | Fichier | Score | Accuracy test | Limite principale |
|---|---|---|---|---:|---|
| Perceptron v1 | Perceptron — bâtiments 3 classes | `buildings_3classes_32x32_perceptron_v1.model` | marge | 52,16 % | modèle linéaire, nombreuses confusions entre styles |
| MLP v2 | MLP — bâtiments 3 classes | `buildings_3classes_32x32_mlp_v2.model` | meilleure probabilité one-vs-rest | 56,76 % | performance encore modeste et scores non calibrés entre modèles |
| RBF v1 | RBF — bâtiments 3 classes | `buildings_3classes_32x32_rbf_v1.model` | meilleure probabilité one-vs-rest | 32,16 % | prédit Gothique pour tous les exemples du test |
| SVM v1 | SVM — bâtiments 3 classes | `buildings_3classes_32x32_svm_v1.model` | marge | 50,81 % | frontière linéaire insuffisante pour de nombreuses images |

**MLP v2 est actuellement le modèle recommandé.** Perceptron v1 et SVM v1
restent des alternatives utiles pour comparer les méthodes linéaires. RBF v1
est conservé pour la comparaison académique, mais il n’est pas fiable sur ce
dataset.

Le mot « confidence » ne doit pas être utilisé :

- Perceptron et SVM retournent une **marge** ;
- MLP et RBF retournent la **meilleure probabilité one-vs-rest** ;
- les scores de deux modèles différents ne sont pas comparables.

# 6. Lancer le projet sous Windows / CLion

Toutes les commandes suivantes sont exécutées depuis :

```text
C:\Users\33783\OneDrive\Documents\clean_proto_lib\Projet-Annuel-3e-ann-e\projet_rendu_2_ml
```

Déclarer les outils exacts de CLion :

```powershell
$env:PATH = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\mingw\bin;$env:PATH"
$cmake = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\cmake\win\x64\bin\cmake.exe"
```

L’environnement web validé utilise :

- Python 3.13.3 ;
- Flask 3.1.3 ;
- Pillow 12.3.0 ;
- Werkzeug 3.1.8.

Installer exactement ces versions avec :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\web\requirements.txt
```

Compiler les principales cibles :

```powershell
& $cmake --build .\cmake-build-debug `
  --target ml_algorithms ml_core c_support demo test_models train_cli tune_cli predict_cli server client `
  -j 1
```

Vérifier ensuite le build parallèle :

```powershell
& $cmake --build .\cmake-build-debug -j 10
```

Lancer le menu principal :

```powershell
.\cmake-build-debug\demo.exe
```

Lancer les tests C++ :

```powershell
.\cmake-build-debug\ml_library_build\test_models.exe
.\cmake-build-debug\demo.exe --tests
.\cmake-build-debug\mlp_xor_audit.exe
```

Tester `predict_cli` avec un fichier contenant exactement 1 024 valeurs
normalisées :

```powershell
.\cmake-build-debug\predict_cli.exe `
  .\models\buildings_3classes_32x32_mlp_v2.model `
  .\chemin\vers\features.txt `
  --expected-class-count 3
```

Préparer puis lancer Flask :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\web\requirements.txt
python .\web\app.py
```

Ouvrir ensuite <http://127.0.0.1:5000>.

# 7. Utiliser demo.exe

`demo.exe` est le principal outil interactif pour expérimenter manuellement.
Depuis le menu :

1. choisir `2. Charger un dataset CSV` ;
2. saisir `data\batiments_3_classes.csv` ;
3. choisir l’algorithme avec les options 4 à 7 ;
4. saisir le nombre d’époques, le learning rate et les paramètres spécifiques ;
5. utiliser l’option 9 pour afficher les accuracies train et test ;
6. utiliser l’option 10 pour sauvegarder le modèle courant ;
7. utiliser l’option 11 pour recharger un modèle sauvegardé ;
8. utiliser l’option 12 pour saisir les caractéristiques et prédire une classe.

La prédiction manuelle demande une valeur pour chaque caractéristique du
modèle. Avec une image 32 × 32, cela représente 1 024 valeurs ; l’interface web
est donc plus pratique pour tester une vraie image.

Un modèle sauvegardé avec `demo.exe` n’apparaît pas automatiquement dans le
site. Pour être proposé par Flask, il doit :

- réellement accepter 1 024 caractéristiques ;
- avoir été entraîné avec les trois classes 0, 1 et 2 ;
- être sauvegardé dans un fichier au nom clair, directement sous `models` ;
- posséder une entrée cohérente dans `models_manifest.json` avec un identifiant
  unique, le fichier, le nom affiché, les dimensions et le type de score.

`demo.exe` seul peut créer le fichier du modèle, mais il ne suffit pas à le
rendre visible sur le site : il ne crée ni rapport officiel ni entrée de
manifest.

# 8. Régénérer les modèles officiels

Pour recréer exactement le catalogue officiel courant et son manifest :

```powershell
.\.venv\Scripts\python.exe .\scripts\train_all_models.py `
  --csv .\data\batiments_3_classes.csv `
  --models-dir .\models `
  --train-cli .\cmake-build-debug\train_cli.exe `
  --seed 42
```

Le catalogue produit contient Perceptron v1, MLP v2, RBF v1 et SVM v1. Le
script appelle quatre fois `train_cli.exe`. L’entraînement, l’évaluation,
la sauvegarde et le rechargement sont donc réalisés par le vrai code C++ via
`ml_api.cpp` et `ml_library`. Python orchestre les commandes, vérifie les
rapports et écrit le JSON du manifest ; il ne réalise aucun apprentissage.

Utiliser cette commande pour changer de dataset, recréer les références,
modifier officiellement leurs paramètres ou vérifier leur reproductibilité.
Ne pas l’utiliser pour démarrer Flask, effectuer une prédiction ou simplement
tester un modèle déjà sauvegardé.

# 9. Tests

Les tests principaux sont :

| Test | Commande ou fichier | Rôle |
|---|---|---|
| Bibliothèque C++ | `cmake-build-debug\ml_library_build\test_models.exe` | vérifie directement les quatre algorithmes |
| API et persistance | `demo.exe --tests` | teste jeux linéaires, XOR, trois classes, sauvegarde et rechargement |
| Flask | `.\.venv\Scripts\python.exe .\web\test_app.py` | teste prétraitement, manifest, image réelle et erreurs web |
| Modèle absent | `web/test_app.py` | vérifie le message lorsqu’un fichier du manifest manque |
| Upload multiple | `web/test_app.py` | vérifie plusieurs images et le nettoyage des temporaires |
| Audit MLP/XOR | `mlp_xor_audit.exe` | teste plusieurs seeds et ordres avec assertions |

Commande Flask :

```powershell
.\.venv\Scripts\python.exe .\web\test_app.py
```

Les cas artificiels affichant une accuracy de `1.000` utilisent les mêmes
petites données pour l’entraînement et l’évaluation. Ils vérifient le câblage de
l’algorithme et la persistance, mais ne mesurent pas sa capacité à généraliser
sur de nouvelles images.

# 10. Limites actuelles

- MLP v2 atteint environ 56,76 % sur le split test actuel : il reste loin d’un
  classifieur parfaitement fiable.
- Les images sont réduites à 32 × 32 pixels et converties en niveaux de gris ;
  beaucoup de détails architecturaux et toutes les informations de couleur
  sont perdus.
- Art déco et Art nouveau restent fréquemment confondus.
- RBF v1 est insuffisant : il prédit Gothique pour tout le jeu de test.
- Le split est reproductible mais non stratifié.
- Une prédiction correcte dans l’interface ne garantit pas que le modèle soit
  fiable sur une autre image ou un autre dataset.
- La meilleure probabilité one-vs-rest n’est pas une garantie de vérité et ne
  doit pas être présentée comme une « confiance » universelle.

# 11. Git et fichiers lourds

| Catégorie | Fichiers concernés | Recommandation |
|---|---|---|
| À versionner | sources C/C++, headers, scripts Python, Flask, templates, tests, `README.md`, `WEB_GUIDE.md`, CMake | oui |
| À versionner avec le modèle | `models_manifest.json`, rapports JSON et modèles officiellement distribués | oui si l’équipe veut que le site fonctionne immédiatement après clonage |
| À ignorer | `cmake-build-debug`, `build`, `.venv`, `__pycache__`, fichiers temporaires et logs | toujours |
| À conserver localement | dataset source, modèles expérimentaux ou obsolètes | recommandé |
| Git LFS éventuel | `data/batiments_3_classes.csv`, dataset source si son partage est indispensable | utiliser LFS plutôt qu’un commit Git classique |

Le dataset source et le CSV ne doivent jamais être ajoutés, commités ou poussés
automatiquement. Les modèles officiels doivent être accompagnés du manifest et
des rapports correspondant exactement à leur version.
