@echo off
set VENV_DIR=.venv

echo Checking for virtual environment in %VENV_DIR%...

if not exist "%VENV_DIR%\Scripts\activate.bat" (
echo Virtual environment not found in %VENV_DIR%.
echo Please make sure the virtual environment exists and the path is correct in the script.
pause
exit /b 1
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

echo Starting GUI_annotator.py...
python GUI_annotator.py

echo GUI closed.
pause