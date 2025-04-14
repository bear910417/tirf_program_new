@echo off
cd .
python -m pip install --upgrade pip
python -m pip install virtualenv
python -m virtualenv .venv
call .venv\Scripts\activate.bat
python -m pip install -v -r requirements.txt
PAUSE