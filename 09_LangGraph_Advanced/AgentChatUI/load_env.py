# setup keys
import getpass
import os
from dotenv import load_dotenv
from pathlib import Path

env_name = "UI"  
env_path = Path("envs") / f"{env_name}.env"

load_dotenv(dotenv_path=env_path)

def set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")