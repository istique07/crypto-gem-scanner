
@echo off
echo Running AI GEM Scanner...
cd /d "%~dp0"
call venv\Scripts\activate.bat
streamlit run main.py
pause
