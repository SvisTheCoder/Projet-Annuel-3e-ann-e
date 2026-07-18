@echo off
setlocal EnableExtensions

set "PROJECT_DIR=%~dp0"
set "BUILD_DIR=%PROJECT_DIR%..\..\..\cmake-build-debug"
set "CMAKE_EXE=C:\Program Files\JetBrains\CLion 2025.2.2\bin\cmake\win\x64\bin\cmake.exe"
set "MINGW_BIN=C:\Program Files\JetBrains\CLion 2025.2.2\bin\mingw\bin"
set "PATH=%MINGW_BIN%;%PATH%"

if not exist "%CMAKE_EXE%" echo ERREUR : CMake CLion est introuvable. & pause & exit /b 1
if not exist "%BUILD_DIR%" echo ERREUR : Lance d'abord un build dans CLion. & pause & exit /b 1

:menu
cls
echo ============================================================
echo           PROJET ML - MENU WINDOWS
echo ============================================================
echo(
echo 1. Compiler tout le projet          5. Lancer la demo interactive
echo 2. Compiler uniquement la lib ML    6. Comparer les 4 modeles sur le CSV
echo 3. Lancer les tests applicatifs
echo 4. Lancer les tests de la lib
echo 0. Quitter
echo(
set choice=
set /p choice="Choisis une option : "

if "%choice%"=="0" goto end
for %%i in (1 2 3 4 5 6) do if "%choice%"=="%%i" goto opt_%%i
echo Choix invalide. & pause & goto menu

:opt_1
echo( & echo Compilation complete du projet...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --clean-first
goto build_done

:opt_2
echo( & echo Compilation de ml_algorithms uniquement...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --target ml_algorithms --clean-first
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
cd /d "%PROJECT_DIR%" && "%BUILD_DIR%\demo.exe" --compare data\batiments_3_classes.csv
goto action_done

:action_done
echo( & echo Code retour : %ERRORLEVEL%
pause & goto menu

:end
endlocal
exit /b 0
