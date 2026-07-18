# Commandes des démonstrations — terminal PowerShell de CLion

## 0. À exécuter dans chaque nouveau terminal CLion

```powershell
$repo = "C:\Users\33783\OneDrive\Documents\clean_proto_lib\Projet-Annuel-3e-ann-e"
$app = "$repo\notebook_modeles_entraines_batiments_clean\app_final\projet_rendu_2_ml"
$build = "$repo\cmake-build-debug"
$python = "$app\.venv\Scripts\python.exe"
$cmake = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\cmake\win\x64\bin\cmake.exe"
$env:PATH = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\mingw\bin;$build;$env:PATH"
```

```powershell
Test-Path "$build\train_cli.exe"
Test-Path "$build\predict_cli.exe"
Test-Path "$app\data\batiments_3_classes.csv"
Test-Path "$app\models\models_manifest.json"
Test-Path $python
```

Résultat :

```text
True
True
True
True
True
```

Si `train_cli.exe` ou `predict_cli.exe` manque :

```powershell
& $cmake --build $build --target train_cli predict_cli -j 1
```

## 1. Slide 23 — entraîner le Perceptron

```powershell
$modeleDemo = "$env:TEMP\soutenance_perceptron.model"
$rapportDemo = "$env:TEMP\soutenance_perceptron.json"
Remove-Item $modeleDemo, $rapportDemo -Force -ErrorAction SilentlyContinue

& "$build\train_cli.exe" `
  --model perceptron `
  --csv "$app\data\batiments_3_classes.csv" `
  --output $modeleDemo `
  --report $rapportDemo `
  --epochs 12 `
  --learning-rate 0.01 `
  --seed 42
```

Résultats attendus :

```text
Nombre de lignes : 1852
Nombre de colonnes : 1025
Nombre de features : 1024
Labels présents : 0 1 2
Train size : 1482
Test size : 370
Train accuracy : 0.6707
Test accuracy : 0.5216
Reload accuracy : 0.5216
```

```powershell
Get-Content $rapportDemo
Get-Item $modeleDemo, $rapportDemo | Select-Object Name, Length
```

Nettoyage :

```powershell
Remove-Item $modeleDemo, $rapportDemo -Force -ErrorAction SilentlyContinue
```

Secours :

```powershell
Get-Content "$app\models\reports\perceptron_v1.json"
```

## 2. Slide 24 — lancer l'application web

### Terminal CLion 1

```powershell
$repo = "C:\Users\33783\OneDrive\Documents\clean_proto_lib\Projet-Annuel-3e-ann-e"
$app = "$repo\notebook_modeles_entraines_batiments_clean\app_final\projet_rendu_2_ml"
$build = "$repo\cmake-build-debug"
$python = "$app\.venv\Scripts\python.exe"
$env:PATH = "C:\Program Files\JetBrains\CLion 2025.2.2\bin\mingw\bin;$build;$env:PATH"
$env:PREDICT_CLI_PATH = "$build\predict_cli.exe"

Set-Location $app
& $python .\web\app.py
```

Résultat :

```text
Running on http://127.0.0.1:5000
```

### Terminal CLion 2

```powershell
Start-Process "http://127.0.0.1:5000"
```

Choisir :

```text
MLP — bâtiments 3 classes
```

Images vérifiées :

```text
C:\Users\33783\OneDrive\Documents\clean_proto_lib\Projet-Annuel-3e-ann-e\projet_rendu_2_ml\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon\Art déco\001946.jpg

C:\Users\33783\OneDrive\Documents\clean_proto_lib\Projet-Annuel-3e-ann-e\projet_rendu_2_ml\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon\Art nouveau\0_655px-Tenement_2C_ca._1910_by_arch._R._Grynus-Gajewski_2C_4_Debnicki_Market_Square_2C_Debniki_2C_Krakow_2C_Poland.jpg

C:\Users\33783\OneDrive\Documents\clean_proto_lib\Projet-Annuel-3e-ann-e\projet_rendu_2_ml\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon\Gothique\008634.jpg
```

Résultats attendus :

```text
001946.jpg                                      -> Art déco
0_655px-Tenement_..._Poland.jpg                 -> Art nouveau
008634.jpg                                      -> Gothique
score_type                                      -> best_ovr_probability
```

Arrêter Flask dans le terminal 1 :

```text
Ctrl+C
```

## 3. Ordre pendant l'oral

```text
SLIDE 23
1. Coller le bloc d'initialisation.
2. Lancer train_cli.
3. Montrer 1852 lignes, 1024 features et les labels 0/1/2.
4. Montrer train accuracy, test accuracy et reload accuracy.
5. Afficher le rapport JSON.

SLIDE 24
1. Lancer Flask dans le terminal 1.
2. Ouvrir la page depuis le terminal 2.
3. Choisir MLP v2.
4. Envoyer les trois images.
5. Montrer les trois classes et score_type.
6. Arrêter Flask avec Ctrl+C.
```

## 4. Tests rapides avant de partir à l'oral

Coller d'abord le bloc de la section 0.

```powershell
& $cmake --build $build --target train_cli predict_cli demo test_models -j 1
```

```powershell
& "$build\ml_library_build\test_models.exe"
```

```powershell
Set-Location $app
& "$build\demo.exe" --tests
```

```powershell
$env:PREDICT_CLI_PATH = "$build\predict_cli.exe"
& $python .\web\test_app.py
```

Résultat final attendu :

```text
ninja: no work to do.
TEST_MODELS : code 0
DEMO --tests : rechargement OK pour tous les cas
WEB : Ran 6 tests
OK
```

## 5. Tests effectués le 18 juillet 2026

| Test | Résultat |
|---|---|
| 5 chemins de préparation | 5 × `True` |
| CMake CLion + Ninja | code 0 |
| `test_models.exe` | code 0 |
| `demo.exe --tests` | code 0, rechargements OK |
| CSV | 1 852 lignes, 1 024 features, labels 0/1/2 |
| `train_cli` Perceptron | code 0, environ 4,3 s |
| Accuracy Perceptron | train 0,6707 ; test 0,5216 ; reload 0,5216 |
| Modèle temporaire | créé, rechargé puis supprimé |
| Rapport temporaire | créé, JSON valide puis supprimé |
| Prétraitement Art déco | 1 024 valeurs dans `[0,1]` |
| Prétraitement Art nouveau | 1 024 valeurs dans `[0,1]` |
| Prétraitement Gothique | 1 024 valeurs dans `[0,1]` |
| MLP v2 sur Art déco | classe 0 correcte |
| MLP v2 sur Art nouveau | classe 1 correcte |
| MLP v2 sur Gothique | classe 2 correcte |
| Fichier de 1 023 features | refusé, code 4 |
| Modèle absent | refusé, code 3 |
| Tests Flask | 6/6 OK en 3,8 s |
| Serveur Flask réel | GET 200, POST multiple 200 |
| Encodage des classes | Art déco / Art nouveau / Gothique OK |
| Nettoyage Flask | aucun dossier `ml_web_*` restant |
| Arrêt Flask | port 5000 libéré |
| Modèles officiels | empreintes identiques avant/après |

Les `Accuracy : 1` de `test_models.exe` signifient 100 % sur de petits cas synthétiques. Pour l'oral, utiliser surtout la sortie plus claire de `demo.exe --tests`.
