"""
conftest.py: pytest runs this file BEFORE the tests. We use it to set up a safe test
environment.

    1. Switch BOTH LLMs off (Groq and Ollama) -> tests are fast, free, and give the same
       result every time. (load_dotenv() never overwrites a variable that already exists,
       so the empty values here win over whatever is in your .env file.)
    2. Use a separate SQLite file for request logs, so tests never touch real logs.
"""

import os
import tempfile

os.environ["LLM_API_KEY"] = ""
os.environ["FALLBACK_LLM_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.gettempdir(), "ivr_test_log.db").replace("\\", "/")
