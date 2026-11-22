"""
Step 0: Check that the setup works
==================================

What this script does (in simple words):
    1. Checks that we are running inside our virtual environment (.venv).
    2. Imports each library we installed and prints its version.
    3. Logs one tiny "hello" run to MLflow, so `mlflow ui` has something to show.

How to run (from inside the ivr-reference folder, with the venv switched on):
    python src/00_check_setup.py

Definitions:
    - Virtual environment (venv): a private folder of Python libraries for ONE project.
      Everyday example: like each project having its own toolbox. This project can use
      pandas 3.x while an older project on the same laptop uses pandas 1.x, and they
      never get in each other's way.
    - Library version: the exact release of a library (e.g. 3.1.1). We print it because
      "it works on my laptop but not on yours" problems often come from different versions.
    - MLflow: a tool that records every experiment we run (the settings we chose and the
      results we got), like a lab notebook that fills itself in.
"""

import sys

# ---------------------------------------------------------------------------
# 1. Are we inside the virtual environment?
# ---------------------------------------------------------------------------
# sys.prefix      = the folder of the Python that is running RIGHT NOW
# sys.base_prefix = the folder of the "main" Python installed on the laptop
# Inside a venv these two are different. Outside a venv they are the same.
#
# WHY check this: the most common setup mistake is installing libraries into one
# Python and then running the script with a different Python. The result is the
# confusing error "ModuleNotFoundError: No module named 'pandas'".
inside_venv: bool = sys.prefix != sys.base_prefix
python_version: str = sys.version.split()[0]   # "3.11.9 (tags/...)" -> "3.11.9"

print("=" * 60)
print("STEP 0: SETUP CHECK")
print("=" * 60)
print(f"Python version      : {python_version}")
print(f"Python running from : {sys.executable}")
print(f"Inside a venv?      : {inside_venv}")

if not inside_venv:
    print()
    print("PROBLEM: you are NOT inside the virtual environment.")
    print("Fix:     .venv\\Scripts\\Activate.ps1   (then run this script again)")
    # sys.exit stops the script here. Exit code 1 means "finished with a problem".
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Import each library and print its version
# ---------------------------------------------------------------------------
# Normally ALL imports go at the top of a file. Here we import them a bit later on
# purpose: if we are outside the venv, the clear message above prints first, instead
# of a scary ModuleNotFoundError.
#
# If an import below fails with "ModuleNotFoundError: No module named 'x'", it means
# library x is not installed in THIS venv. Fix: pip install -r requirements-dev.txt
import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # hide an MLflow hint meant for AI assistants

import numpy        # fast maths on arrays (lists of numbers)
import pandas       # tables (DataFrames), like Excel inside Python
import sklearn      # scikit-learn: classic ML models + metrics (pip name: scikit-learn)
import matplotlib   # charts and plots
import mlflow       # experiment tracking

print()
print("Library versions:")
print(f"  numpy        : {numpy.__version__}")
print(f"  pandas       : {pandas.__version__}")
print(f"  scikit-learn : {sklearn.__version__}")
print(f"  matplotlib   : {matplotlib.__version__}")
print(f"  mlflow       : {mlflow.__version__}")

# ---------------------------------------------------------------------------
# 3. Log one tiny test run to MLflow
# ---------------------------------------------------------------------------
# MLflow words (we will use these in every step from Step 3 onwards):
#   - experiment = a group of related runs, like a folder (e.g. "step3-tfidf-baseline")
#   - run        = ONE execution of a training script
#   - param      = a setting we CHOSE before training      (e.g. max_features = 5000)
#   - metric     = a number we MEASURED after training     (e.g. macro_f1 = 0.87)
#
# Where does MLflow save this? MLflow 3.x saves it in a file called mlflow.db, inside
# the folder you ran the script FROM. That is why we always run scripts from the
# ivr-reference folder: then `mlflow ui`, started from the same folder, finds the runs.
print()
print("MLflow: logging a test run...")

# set_experiment: use this experiment (MLflow creates it the first time).
mlflow.set_experiment("step0-setup-check")

# "with" starts a run and AUTOMATICALLY ends it when the indented block finishes,
# even if an error happens inside. (Without "with", we would have to remember to
# call mlflow.end_run() ourselves.)
with mlflow.start_run(run_name="hello-mlflow"):
    mlflow.log_param("python_version", python_version)
    mlflow.log_metric("setup_ok", 1.0)

print("MLflow: done. Experiment 'step0-setup-check' now has a run called 'hello-mlflow'.")

print()
print("=" * 60)
print("SETUP OK")
print("Next: run   mlflow ui   and open http://127.0.0.1:5000 in your browser.")
print("=" * 60)
