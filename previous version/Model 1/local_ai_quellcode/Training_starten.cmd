@echo off
setlocal
cd /d "%~dp0local_ai"
set "AI_PYTHON=%~dp0..\work\.venv\Scripts\python.exe"
if exist ".venv\Scripts\python.exe" set "AI_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%AI_PYTHON%" (
    echo Python-Umgebung fehlt. Siehe local_ai\README.md.
    pause
    exit /b 1
)
echo Echtes Training mit eigenen Modellgewichten, zunaechst bis zu zwei Stunden.
echo Datenaufbereitung kann zusaetzliche Zeit benoetigen.
echo Zum sicheren Stoppen Training_stoppen.cmd verwenden.
"%AI_PYTHON%" -u learn.py --hours 2 --resume
set "AI_EXIT_CODE=%ERRORLEVEL%"
if not "%AI_EXIT_CODE%"=="0" pause
exit /b %AI_EXIT_CODE%
