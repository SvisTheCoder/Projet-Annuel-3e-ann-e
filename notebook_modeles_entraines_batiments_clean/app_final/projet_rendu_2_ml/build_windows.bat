@echo off
setlocal EnableExtensions

REM ============================================================
REM Projet ML - Commandes de build et lancement UNIVERSAL
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build"
set "CMAKE_EXE=cmake"

REM --- 1. VERIFICATION DE CMAKE ---
where %CMAKE_EXE% >nul 2>nul
if not errorlevel 1 goto cmake_found
echo ERREUR : CMake est introuvable dans le PATH de votre systeme.
echo Veuillez installer CMake et l'ajouter aux variables d'environnement (PATH).
pause
exit /b 1

:cmake_found

REM --- 2. CONFIGURATION ET GENERATION AUTOMATIQUE DU DOSSIER BUILD ---
if exist "%BUILD_DIR%" goto build_dir_exists
echo Le dossier de build est introuvable. Creation et configuration initiale avec CMake...
mkdir "%BUILD_DIR%"
"%CMAKE_EXE%" -S "%FORWARD_PROJECT_DIR%" -B "%FORWARD_BUILD_DIR%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug
REM Remplacement des antislashs par des slashs pour eviter que le \ final n'echappe les guillemets
set "FORWARD_PROJECT_DIR=%PROJECT_DIR:\=/%"
set "FORWARD_BUILD_DIR=%BUILD_DIR:\=/%"

"%CMAKE_EXE%" -S "%FORWARD_PROJECT_DIR%" -B "%FORWARD_BUILD_DIR%" -DCMAKE_BUILD_TYPE=Debug
if not errorlevel 1 goto build_dir_exists
echo ERREUR : La configuration de CMake a echoue. 
echo Verifiez qu'un compilateur (MinGW/GCC ou MSVC) est bien installe et accessible.
pause
exit /b 1

:build_dir_exists

:menu
cls
echo ============================================================
echo           PROJET ML - MENU WINDOWS (STANDALONE)
echo ============================================================
echo(
echo 1. Compiler tout le projet
echo 2. Compiler uniquement la librairie ML
echo 3. Lancer les tests applicatifs
echo 4. Lancer les tests de la librairie
echo 5. Lancer la demo interactive
echo 6. Comparer les 4 modeles sur le CSV
echo 7. Lancer le serveur TCP
echo 8. Lancer le client TCP
echo 0. Quitter
echo(
set choice=
set /p choice="Choisis une option : "

if "%choice%"=="1" goto build_all
if "%choice%"=="2" goto build_library
if "%choice%"=="3" goto run_tests
if "%choice%"=="4" goto run_library_tests
if "%choice%"=="5" goto run_demo
if "%choice%"=="6" goto compare_models
if "%choice%"=="7" goto run_server
if "%choice%"=="8" goto run_client
if "%choice%"=="0" goto end

echo Choix invalide.
pause
goto menu


:build_all
echo(
echo Compilation complete du projet...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Debug --clean-first
if errorlevel 1 goto build_all_failed
echo(
echo Compilation terminee avec succes.
goto build_all_end
:build_all_failed
echo(
echo ECHEC DE LA COMPILATION.
:build_all_end
pause
goto menu


:build_library
echo(
echo Compilation de ml_algorithms uniquement...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --target ml_algorithms --config Debug --clean-first
if errorlevel 1 goto build_library_failed
echo(
echo Librairie compilee avec succes.
goto build_library_end
:build_library_failed
echo(
echo ECHEC DE LA COMPILATION DE LA LIBRAIRIE.
:build_library_end
pause
goto menu


REM --- FONCTION INTERNE POUR REPERER LES EXECUTABLES ---
:find_bin_dir
set "BIN_DIR=%BUILD_DIR%"
if exist "%BUILD_DIR%\Debug\demo.exe" set "BIN_DIR=%BUILD_DIR%\Debug"
goto :eof


:run_tests
echo(
echo Lancement des tests applicatifs...
call :find_bin_dir
cd /d "%BIN_DIR%"
demo.exe --tests
echo(
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_library_tests
echo(
echo Lancement des tests de ml_library...
set "LIB_TEST_PATH=%BUILD_DIR%\ml_library_build\test_models.exe"
if exist "%BUILD_DIR%\ml_library_build\Debug\test_models.exe" set "LIB_TEST_PATH=%BUILD_DIR%\ml_library_build\Debug\test_models.exe"
"%LIB_TEST_PATH%"
echo(
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_demo
echo(
echo Lancement de la demo interactive...
echo Pour charger les donnees, utilise : data\batiments.csv
echo(
call :find_bin_dir
cd /d "%PROJECT_DIR%"
"%BIN_DIR%\demo.exe"
echo(
echo Code retour : %ERRORLEVEL%
pause
goto menu


:compare_models
echo(
echo Comparaison des quatre modeles sur le CSV...
echo Cette operation peut etre longue.
echo(
call :find_bin_dir
cd /d "%PROJECT_DIR%"
"%BIN_DIR%\demo.exe" --compare data\batiments.csv
echo(
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_server
echo(
set /p model_path="Chemin du modele [build\validation_perceptron.model] : "
if "%model_path%"=="" set "model_path=build\validation_perceptron.model"

set /p port="Port [54321] : "
if "%port%"=="" set "port=54321"

echo(
echo Lancement du serveur...
echo Modele : %model_path%
echo Port   : %port%
echo(
call :find_bin_dir
cd /d "%PROJECT_DIR%"
"%BIN_DIR%\server.exe" "%model_path%" %port%
echo(
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_client
echo(
set /p host="Adresse serveur [127.0.0.1] : "
if "%host%"=="" set "host=127.0.0.1"

set /p port="Port [54321] : "
if "%port%"=="" set "port=54321"

set /p features="Nombre de caracteristiques [1024] : "
if "%features%"=="" set "features=1024"

echo(
echo Lancement du client...
echo Le client demandera %features% valeurs.
echo(
call :find_bin_dir
cd /d "%PROJECT_DIR%"
"%BIN_DIR%\client.exe" %host% %port% %features%
echo(
echo Code retour : %ERRORLEVEL%
pause
goto menu


:end
endlocal
exit /b 0