# Interface web locale

Pour la documentation générale, l’architecture et le cycle de vie des modèles,
consultez aussi le [README principal](README.md).

Cette interface reste séparée de `demo`. Python gère seulement la page web,
l’upload et le prétraitement des images. La prédiction est toujours faite par
`predict_cli`, `ml_api.cpp` et les modèles C++ de `ml_library`.

Toutes les commandes ci-dessous sont lancées depuis `projet_rendu_2_ml`.

## 1. Préparer Python

Versions validées : Python 3.13.3, Flask 3.1.3, Pillow 12.3.0 et Werkzeug
3.1.8. Le fichier `requirements.txt` fixe exactement ces versions.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\web\requirements.txt
```

## 2. Générer le CSV canonique

```powershell
python .\scripts\images_to_csv.py `
  ".\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon" `
  ".\data\batiments_3_classes.csv"
```

Le script affiche 1 852 lignes, 1 025 colonnes, 1 024 features et la
distribution 554 / 745 / 553 des labels 0 / 1 / 2. Les fichiers sont triés,
donc deux exécutions sur le même dataset produisent le même CSV.

## 3. Compiler avec la toolchain CLion

```powershell
$env:PATH = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\mingw\bin;$env:PATH"
$cmake = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\cmake\win\x64\bin\cmake.exe"
& $cmake --build .\cmake-build-debug --target train_cli predict_cli demo test_models server client -j 1
```

## 4. Entraîner les quatre modèles et créer le manifest

```powershell
python .\scripts\train_all_models.py `
  --csv .\data\batiments_3_classes.csv `
  --models-dir .\models `
  --train-cli .\cmake-build-debug\train_cli.exe
```

Cette commande recrée exactement Perceptron v1, MLP v2, RBF v1 et SVM v1,
leurs rapports dans `models/reports` et `models/models_manifest.json`. Pour MLP
v2, elle applique 12 epochs, un learning rate de 0,02 et 32 neurones cachés.
Chaque modèle est sauvegardé, rechargé puis contrôlé avant d’être ajouté au
manifest.

Pour entraîner un seul modèle, il faut aussi fournir son rapport :

```powershell
.\cmake-build-debug\train_cli.exe `
  --model perceptron `
  --csv .\data\batiments_3_classes.csv `
  --output .\models\buildings_3classes_32x32_perceptron_v1.model `
  --report .\models\reports\perceptron_v1.json `
  --seed 42
```

## 5. Lancer Flask

```powershell
python .\web\app.py
```

Ouvrir ensuite <http://127.0.0.1:5000>. L’interface lit le manifest et
n’affiche que les modèles compatibles réellement présents.

Elle accepte au maximum 10 images PNG/JPG/JPEG de 10 Mo chacune. Les scores
sont des marges pour Perceptron/SVM et les meilleures probabilités one-vs-rest
pour MLP/RBF. Ils ne sont comparables qu’à l’intérieur d’un même modèle.

## 6. Tester predict_cli directement

Le deuxième argument est un fichier texte contenant exactement 1 024 valeurs
normalisées entre 0 et 1 :

```powershell
.\cmake-build-debug\predict_cli.exe `
  .\models\buildings_3classes_32x32_perceptron_v1.model `
  .\chemin\vers\features.txt `
  --expected-class-count 3
```

La sortie standard contient uniquement le JSON. Les erreurs sont écrites sur
la sortie d’erreur avec un code retour non nul.
