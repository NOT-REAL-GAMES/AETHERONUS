@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "CONFIG=%~1"
if "%CONFIG%"=="" set "CONFIG=Debug"

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "BUILD_DIR=%ROOT%\build"
set "EXE_FOUND="
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

cmake -S "%ROOT%" -B "%BUILD_DIR%"
if errorlevel 1 (
    echo CMake configure failed.
    exit /b 1
)

cmake --build "%BUILD_DIR%" --config "%CONFIG%" --parallel
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

set "EXE=%BUILD_DIR%\%CONFIG%\AETHERONUS.exe"
if not exist "%EXE%" set "EXE=%BUILD_DIR%\x64\%CONFIG%\AETHERONUS.exe"
if not exist "%EXE%" set "EXE=%BUILD_DIR%\bin\%CONFIG%\AETHERONUS.exe"

if not exist "%EXE%" for /R "%BUILD_DIR%" %%F in (AETHERONUS.exe) do (
    echo %%~dpF | findstr /I /C:"\%CONFIG%\\" >nul
    if not errorlevel 1 if not defined EXE_FOUND set "EXE_FOUND=%%F"
)

if defined EXE_FOUND set "EXE=%EXE_FOUND%"

if not exist "%EXE%" (
    echo Could not find AETHERONUS.exe in "%BUILD_DIR%".
    exit /b 1
)

echo Launching %EXE%
pushd "%ROOT%"
"%EXE%"
set "RUN_RESULT=%ERRORLEVEL%"
popd
endlocal & exit /b %RUN_RESULT%
