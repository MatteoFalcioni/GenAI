# agent_ui/load_env.py

import getpass
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path("agent_ui") / f".env"

# Load dotenv file right away
load_dotenv(dotenv_path=env_path, override=False)

def set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

# Ensure these are loaded unless already defined
set_if_undefined("OPENAI_API_KEY")
set_if_undefined("LANGSMITH_API_KEY")
set_if_undefined("LANGSMITH_TRACING")
set_if_undefined("LANGSMITH_ENDPOINT")
set_if_undefined("LANGSMITH_PROJECT")
