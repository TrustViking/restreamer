@echo off
chcp 65001 >nul
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv_restreamer\Scripts\python.exe"
set "PY_SCRIPT=%SCRIPT_DIR%promo.py"

if not exist "%PYTHON%" (
  echo [ERROR] Python not found: "%PYTHON%"
  pause
  exit /b 1
)

if not exist "%PY_SCRIPT%" (
  echo [ERROR] Script not found: "%PY_SCRIPT%"
  pause
  exit /b 1
)

cd /d "%SCRIPT_DIR%"

echo [INFO] BAT argv: %*
echo [INFO] BAT cwd: %CD%
echo [INFO] BAT python: "%PYTHON%"
echo [INFO] BAT script: "%PY_SCRIPT%"

"%PYTHON%" "%PY_SCRIPT%" %*

echo.
echo [INFO] Exit code: %ERRORLEVEL%
pause
