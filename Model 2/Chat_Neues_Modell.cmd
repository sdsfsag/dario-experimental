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
echo Dieses Fenster wartet auf den neuen Trainingslauf und startet danach dessen Chat.
echo Strg+C beendet nur das Warten. /quit beendet den Chat.
if exist "runs\improvement_v5\status.json" (
    "%AI_PYTHON%" -u improvement_train.py --chat
) else if exist "runs\dialogue_v4\status.json" (
    "%AI_PYTHON%" -u dialogue_train.py --chat
) else (
    "%AI_PYTHON%" -u quality_train.py --chat
)
set "AI_EXIT_CODE=%ERRORLEVEL%"
if not "%AI_EXIT_CODE%"=="0" pause
exit /b %AI_EXIT_CODE%
