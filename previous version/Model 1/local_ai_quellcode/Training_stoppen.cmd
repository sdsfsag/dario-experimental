@echo off
setlocal
if not exist "%~dp0local_ai\runs\real_v1" (
    echo Noch kein echtes Training eingerichtet.
    pause
    exit /b 1
)
echo stop>"%~dp0local_ai\runs\real_v1\STOP"
echo Stopp angefordert. Der laufende Optimizer-Schritt wird beendet und gespeichert.
echo Waehrend einer Datenaufbereitung wird deren aktueller Arbeitsschritt zuerst abgeschlossen.
pause
