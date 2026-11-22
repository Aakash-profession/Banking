# IVR Intent Router: reference version (private)

This is the **Phase 1 reference** built step by step with Claude Code, so I can learn from it.
It is never pushed to GitHub. My own rebuild lives in `../ivr-intent-router/`.

## Setup (Windows PowerShell, run inside this folder)

```powershell
py -3.11 -m venv .venv                     # create the virtual environment
.venv\Scripts\Activate.ps1                 # switch it on (the prompt shows "(.venv)")
python -m pip install --upgrade pip        # newer pip = fewer install errors
pip install -r requirements-dev.txt        # install all laptop libraries
python src/00_check_setup.py               # check everything works
mlflow ui                                  # open http://127.0.0.1:5000
```

Always run scripts and `mlflow ui` **from this folder**, because MLflow saves runs to
`mlflow.db` in whichever folder you are in.

## Steps

| Step | Script | Status |
|---|---|---|
| 0 | `src/00_check_setup.py` | setup check |
