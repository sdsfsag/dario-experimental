@echo off
setlocal
cd /d "%~dp0local_ai"
set "AI_PYTHON=%~dp0..\work\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "AI_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%AI_PYTHON%" (
    echo Python-Umgebung fehlt. Bitte die Installation in local_ai\README.md ausfuehren.
    pause
    exit /b 1
)
"%AI_PYTHON%" -u start_chat.py %*
set "AI_EXIT_CODE=%ERRORLEVEL%"
if not "%AI_EXIT_CODE%"=="0" (
    echo.
    echo Eine beliebige Taste schliesst nur dieses Fenster.
    pause >nul
)
exit /b %AI_EXIT_CODE%
