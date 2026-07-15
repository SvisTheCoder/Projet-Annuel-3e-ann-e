@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%build"

REM --- 1. VERIFICATION DES OUTILS (CMake et g++) ---
where cmake >nul 2>nul
if errorlevel 1 echo ERREUR : CMake est introuvable dans le PATH. & pause & exit /b 1
where g++ >nul 2>nul
if errorlevel 1 echo ERREUR : g++ (MinGW) est introuvable dans le PATH. & pause & exit /b 1

REM --- 2. CONFIGURATION INITIALE UNIQUE (MinGW Makefiles) ---
if exist "%BUILD_DIR%" goto menu
echo Configuration initiale du projet avec MinGW...
set "FORWARD_PROJECT_DIR=%PROJECT_DIR:\=/%"
set "FORWARD_BUILD_DIR=%BUILD_DIR:\=/%"
mkdir "%BUILD_DIR%"
cmake -S "%FORWARD_PROJECT_DIR%" -B "%FORWARD_BUILD_DIR%" -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug
if errorlevel 1 echo ERREUR : La configuration a echoue. & rmdir /s /q "%BUILD_DIR%" & pause & exit /b 1

:menu
cls
echo ============================================================
echo           PROJET ML - MENU WINDOWS (CMake + g++)
echo ============================================================
echo(
echo 1. Compiler tout le projet          5. Lancer la demo interactive
echo 2. Compiler uniquement la lib ML    6. Comparer les 4 modeles sur le CSV
echo 3. Lancer les tests applicatifs     7. Lancer le serveur TCP
echo 4. Lancer les tests de la lib       8. Lancer le client TCP
echo 0. Quitter
echo(
set choice=
set /p choice="Choisis une option : "

if "%choice%"=="0" goto end
for %%i in (1 2 3 4 5 6 7 8) do if "%choice%"=="%%i" goto opt_%%i
echo Choix invalide. & pause & goto menu

:opt_1
echo( & echo Compilation complete du projet...
cmake --build "%BUILD_DIR%" --clean-first
goto build_done

:opt_2
echo( & echo Compilation de ml_algorithms uniquement...
cmake --build "%BUILD_DIR%" --target ml_algorithms --clean-first
goto build_done

:build_done
if errorlevel 1 (echo( & echo ECHEC DE LA COMPILATION.) else (echo( & echo Succes !)
pause & goto menu

:opt_3
echo( & echo Lancement des tests applicatifs...
cd /d "%BUILD_DIR%" && demo.exe --tests
goto action_done

:opt_4
echo( & echo Lancement des tests de ml_library...
"%BUILD_DIR%\ml_library_build\test_models.exe"
goto action_done

:opt_5
echo( & echo Lancement de la demo interactive...
cd /d "%PROJECT_DIR%" && "%BUILD_DIR%\demo.exe"
goto action_done

:opt_6
echo( & echo Comparaison des quatre modeles sur le CSV...
cd /d "%PROJECT_DIR%" && "%BUILD_DIR%\demo.exe" --compare data\batiments.csv
goto action_done

:opt_7
echo(
set /p model_path="Chemin du modele [build\validation_perceptron.model] : "
if "%model_path%"=="" set "model_path=build\validation_perceptron.model"
set /p port="Port [54321] : "
if "%port%"=="" set "port=54321"
echo( & echo Lancement du serveur...
cd /d "%PROJECT_DIR%" && "%BUILD_DIR%\server.exe" "%model_path%" %port%
goto action_done

:opt_8
echo(
set /p host="Adresse serveur [127.0.0.1] : "
if "%host%"=="" set "host=127.0.0.1"
set /p port="Port [54321] : "
if "%port%"=="" set "port=54321"
set /p features="Nombre de caracteristiques [1024] : "
if "%features%"=="" set "features=1024"
echo( & echo Lancement du client...
cd /d "%PROJECT_DIR%" && "%BUILD_DIR%\client.exe" %host% %port% %features%
goto action_done

:action_done
echo( & echo Code retour : %ERRORLEVEL%
pause & goto menu

:end
endlocal
exit /b 0