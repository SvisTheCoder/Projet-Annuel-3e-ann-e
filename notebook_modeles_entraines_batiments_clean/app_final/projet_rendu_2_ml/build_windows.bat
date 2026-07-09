@echo off
setlocal EnableExtensions

REM ============================================================
REM Projet ML - commandes de build et lancement sous Windows
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%cmake-build-debug"
set "CMAKE_EXE=C:\Users\User\AppData\Local\Programs\CLion\bin\cmake\win\x64\bin\cmake.exe"
set "MINGW_BIN=C:\Program Files\JetBrains\CLion 2025.2.2\bin\mingw\bin"

set "PATH=%MINGW_BIN%;%PATH%"

if not exist "%CMAKE_EXE%" (
    echo ERREUR : CMake CLion introuvable :
    echo %CMAKE_EXE%
    pause
    exit /b 1
)

if not exist "%BUILD_DIR%" (
    echo ERREUR : Le dossier de build est introuvable :
    echo %BUILD_DIR%
    echo Ouvre d'abord le projet dans CLion et lance un build.
    pause
    exit /b 1
)

:menu
cls
echo ============================================================
echo           PROJET ML - MENU WINDOWS
echo ============================================================
echo.
echo 1. Compiler tout le projet
echo 2. Compiler uniquement la librairie ML
echo 3. Lancer les tests applicatifs
echo 4. Lancer les tests de la librairie
echo 5. Lancer la demo interactive
echo 6. Comparer les 4 modeles sur le CSV
echo 7. Lancer le serveur TCP
echo 8. Lancer le client TCP
echo 0. Quitter
echo.
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
echo.
echo Compilation complete du projet...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --clean-first -j 1

if errorlevel 1 (
    echo.
    echo ECHEC DE LA COMPILATION.
) else (
    echo.
    echo Compilation terminee avec succes.
)

pause
goto menu


:build_library
echo.
echo Compilation de ml_algorithms uniquement...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --target ml_algorithms --clean-first -j 1

if errorlevel 1 (
    echo.
    echo ECHEC DE LA COMPILATION DE LA LIBRAIRIE.
) else (
    echo.
    echo Librairie compilee avec succes.
)

pause
goto menu


:run_tests
echo.
echo Lancement des tests applicatifs...
cd /d "%BUILD_DIR%"
demo.exe --tests

echo.
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_library_tests
echo.
echo Lancement des tests de ml_library...
cd /d "%BUILD_DIR%"
ml_library_build\test_models.exe

echo.
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_demo
echo.
echo Lancement de la demo interactive...
echo Pour charger les donnees, utilise : data\batiments.csv
echo.
cd /d "%PROJECT_DIR%"
cmake-build-debug\demo.exe

echo.
echo Code retour : %ERRORLEVEL%
pause
goto menu


:compare_models
echo.
echo Comparaison des quatre modeles sur le CSV...
echo Cette operation peut etre longue.
echo.
cd /d "%PROJECT_DIR%"
cmake-build-debug\demo.exe --compare data\batiments.csv

echo.
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_server
echo.
set /p model_path="Chemin du modele [cmake-build-debug\validation_perceptron.model] : "

if "%model_path%"=="" (
    set "model_path=cmake-build-debug\validation_perceptron.model"
)

set /p port="Port [54321] : "

if "%port%"=="" (
    set "port=54321"
)

echo.
echo Lancement du serveur...
echo Modele : %model_path%
echo Port   : %port%
echo.
cd /d "%PROJECT_DIR%"
cmake-build-debug\server.exe "%model_path%" %port%

echo.
echo Code retour : %ERRORLEVEL%
pause
goto menu


:run_client
echo.
set /p host="Adresse serveur [127.0.0.1] : "

if "%host%"=="" (
    set "host=127.0.0.1"
)

set /p port="Port [54321] : "

if "%port%"=="" (
    set "port=54321"
)

set /p features="Nombre de caracteristiques [1024] : "

if "%features%"=="" (
    set "features=1024"
)

echo.
echo Lancement du client...
echo Le client demandera %features% valeurs.
echo.
cd /d "%PROJECT_DIR%"
cmake-build-debug\client.exe %host% %port% %features%

echo.
echo Code retour : %ERRORLEVEL%
pause
goto menu


:end
endlocal
exit /b 0