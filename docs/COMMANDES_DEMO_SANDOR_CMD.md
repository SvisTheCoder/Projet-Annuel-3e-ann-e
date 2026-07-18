# Démonstration Sandor — CMD

## Configuration unique du PC

```bat
chcp 65001 >nul
set "REPO=C:\Users\sgeno\Desktop\github\Projet-Annuel-3e-ann-e"
set "APP=%REPO%\notebook_modeles_entraines_batiments_clean\app_final\projet_rendu_2_ml"
set "BUILD=%REPO%\cmake-build-sandor-demo"
set "CLION=C:\Program Files\JetBrains\CLion 2025.2.2"
set "CMAKE=%CLION%\bin\cmake\win\x64\bin\cmake.exe"
set "NINJA=%CLION%\bin\ninja\win\x64\ninja.exe"
set "MINGW=%CLION%\bin\mingw\bin"
set "DATASET=%APP%\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon"
if not exist "%DATASET%" set "DATASET=%REPO%\projet_rendu_2_ml\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon"
set "PATH=%MINGW%;%BUILD%;%PATH%"
```

```bat
if exist "%REPO%" (echo OK REPO) else (echo ERREUR REPO)
if exist "%APP%\CMakeLists.txt" (echo OK CMAKE_LISTS) else (echo ERREUR CMAKE_LISTS)
if exist "%CMAKE%" (echo OK CMAKE) else (echo ERREUR CMAKE)
if exist "%NINJA%" (echo OK NINJA) else (echo ERREUR NINJA)
if exist "%MINGW%\g++.exe" (echo OK G++) else (echo ERREUR G++)
if exist "%DATASET%\Art déco\001946.jpg" (echo OK IMAGE ART DECO) else (echo ERREUR IMAGE ART DECO)
if exist "%DATASET%\Art nouveau\0_655px-Tenement_2C_ca._1910_by_arch._R._Grynus-Gajewski_2C_4_Debnicki_Market_Square_2C_Debniki_2C_Krakow_2C_Poland.jpg" (echo OK IMAGE ART NOUVEAU) else (echo ERREUR IMAGE ART NOUVEAU)
if exist "%DATASET%\Gothique\008634.jpg" (echo OK IMAGE GOTHIQUE) else (echo ERREUR IMAGE GOTHIQUE)
py -3 --version
```

```bat
"%CMAKE%" -S "%APP%" -B "%BUILD%" -G Ninja ^
  "-DCMAKE_BUILD_TYPE=Debug" ^
  "-DCMAKE_MAKE_PROGRAM=%NINJA%" ^
  "-DCMAKE_C_COMPILER=%MINGW%\gcc.exe" ^
  "-DCMAKE_CXX_COMPILER=%MINGW%\g++.exe"
```

```bat
"%CMAKE%" --build "%BUILD%" --target train_cli predict_cli demo test_models -j 1
```

```bat
if not exist "%APP%\.venv\Scripts\python.exe" py -3 -m venv "%APP%\.venv"
set "PYTHON=%APP%\.venv\Scripts\python.exe"
"%PYTHON%" -m pip install -r "%APP%\web\requirements.txt"
```

```bat
if exist "%BUILD%\train_cli.exe" (echo OK TRAIN_CLI) else (echo ERREUR TRAIN_CLI)
if exist "%BUILD%\predict_cli.exe" (echo OK PREDICT_CLI) else (echo ERREUR PREDICT_CLI)
if exist "%BUILD%\demo.exe" (echo OK DEMO) else (echo ERREUR DEMO)
if exist "%BUILD%\ml_library_build\test_models.exe" (echo OK TEST_MODELS) else (echo ERREUR TEST_MODELS)
if exist "%APP%\data\batiments_3_classes.csv" (echo OK CSV) else (echo ERREUR CSV)
if exist "%APP%\models\models_manifest.json" (echo OK MANIFEST) else (echo ERREUR MANIFEST)
if exist "%PYTHON%" (echo OK PYTHON) else (echo ERREUR PYTHON)
```

## Bloc à coller au début de chaque nouveau CMD

```bat
chcp 65001 >nul
set "REPO=C:\Users\sgeno\Desktop\github\Projet-Annuel-3e-ann-e"
set "APP=%REPO%\notebook_modeles_entraines_batiments_clean\app_final\projet_rendu_2_ml"
set "BUILD=%REPO%\cmake-build-sandor-demo"
set "CLION=C:\Program Files\JetBrains\CLion 2025.2.2"
set "MINGW=%CLION%\bin\mingw\bin"
set "PYTHON=%APP%\.venv\Scripts\python.exe"
set "DATASET=%APP%\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon"
if not exist "%DATASET%" set "DATASET=%REPO%\projet_rendu_2_ml\data\Dataset final-20260702T055807Z-3-001\Dataset final sans doublon"
set "PATH=%MINGW%;%BUILD%;%PATH%"
cd /d "%APP%"
```

## Diapositive 21 — cas de test

```bat
"%BUILD%\demo.exe" --tests
echo CODE RETOUR = %ERRORLEVEL%
```

Résultat :

```text
rechargement : OK
CODE RETOUR = 0
```

## Diapositive 23 — entraînement

```bat
set "MODELE_DEMO=%TEMP%\soutenance_perceptron.model"
set "RAPPORT_DEMO=%TEMP%\soutenance_perceptron.json"
del /q "%MODELE_DEMO%" "%RAPPORT_DEMO%" 2>nul
```

```bat
"%BUILD%\train_cli.exe" ^
  --model perceptron ^
  --csv "%APP%\data\batiments_3_classes.csv" ^
  --output "%MODELE_DEMO%" ^
  --report "%RAPPORT_DEMO%" ^
  --epochs 12 ^
  --learning-rate 0.01 ^
  --seed 42
```

```bat
echo CODE RETOUR = %ERRORLEVEL%
type "%RAPPORT_DEMO%"
dir "%MODELE_DEMO%" "%RAPPORT_DEMO%"
```

Résultat :

```text
Nombre de lignes : 1852
Nombre de features : 1024
Labels présents : 0 1 2
Train size : 1482
Test size : 370
Train accuracy : 0.6707
Test accuracy : 0.5216
Reload accuracy : 0.5216
CODE RETOUR = 0
```

Nettoyage :

```bat
del /q "%MODELE_DEMO%" "%RAPPORT_DEMO%"
```

## Diapositive 24 — application web

### CMD 1

```bat
set "PREDICT_CLI_PATH=%BUILD%\predict_cli.exe"
cd /d "%APP%"
"%PYTHON%" web\app.py
```

Résultat :

```text
Running on http://127.0.0.1:5000
```

### CMD 2

```bat
start "" http://127.0.0.1:5000
```

Choisir :

```text
MLP — bâtiments 3 classes
```

Images :

```bat
explorer /select,"%DATASET%\Art déco\001946.jpg"
explorer /select,"%DATASET%\Art nouveau\0_655px-Tenement_2C_ca._1910_by_arch._R._Grynus-Gajewski_2C_4_Debnicki_Market_Square_2C_Debniki_2C_Krakow_2C_Poland.jpg"
explorer /select,"%DATASET%\Gothique\008634.jpg"
```

Résultat :

```text
001946.jpg -> Art déco
0_655px-Tenement_..._Poland.jpg -> Art nouveau
008634.jpg -> Gothique
```

Arrêter le serveur dans CMD 1 :

```text
Ctrl+C
```

## Vérification finale

```bat
netstat -ano | findstr ":5000"
```

Résultat après `Ctrl+C` : aucune ligne `LISTENING`.
