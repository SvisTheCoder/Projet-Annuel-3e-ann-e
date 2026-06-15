@echo off
cmake -S . -B build
if errorlevel 1 exit /b 1
cmake --build build --config Release
if errorlevel 1 exit /b 1
echo.
echo Compilation terminee.
echo Lancez build\Release\demo.exe
pause
