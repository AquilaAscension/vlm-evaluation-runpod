import os
from dotenv import load_dotenv

def load_keys():
    if os.path.exists(".keys.env"):
        load_dotenv(".keys.env")
    else:
        raise FileNotFoundError(".keys.env not found — please create it and add your keys.")
