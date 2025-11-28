# Rebuilding the Sphinx Documentation

Note - this is only needed if you want to test changes to the docs locally before pushing them to GitHub.
ReadTheDocs will build the docs automatically.

## 1) Activate your virtualenv
``` bash
# bash
source .venv/bin/activate
```

``` powershell

# PowerShell
. .venv/Scripts/Activate.ps1
```
## 2) Install Sphinx (and optional extensions)
``` bash
pip install sphinx sphinx-autodoc-typehints
```
## 3) Go to the docs folder
``` bash
cd docs
```
## 4) Build the HTML docs
- If you have a Makefile (Linux/macOS):
``` bash
make clean html
```
- If you have make.bat (Windows):
``` powershell
.\make.bat clean html
```
- Or directly with sphinx-build:
``` bash
sphinx-build -b html . _build/html
```

```
## 6) Open the result
``` bash
# Start a webserver
python -m http.server --directory build/html 8000
```
Then open in your browser
http://localhost:8000
```
