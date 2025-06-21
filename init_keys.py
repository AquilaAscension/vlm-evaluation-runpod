#!/usr/bin/env python3
import os, getpass, pathlib, textwrap
ENV_FILE = pathlib.Path(".keys.env")
if ENV_FILE.exists():
    print(".keys.env already exists – nothing to do.")
    exit()

print("Enter your keys (leave blank if not needed).")
anthropic = getpass.getpass("ANTHROPIC_API_KEY: ")
google    = getpass.getpass("GOOGLE_API_KEY: ")
ernie_key = getpass.getpass("ERNIE_API_KEY: ")
ernie_sec = getpass.getpass("ERNIE_SECRET_KEY: ")
hf_token  = getpass.getpass("HF_TOKEN (Hugging Face): ")

content = textwrap.dedent(f"""\
    ANTHROPIC_API_KEY={anthropic}
    GOOGLE_API_KEY={google}
    ERNIE_API_KEY={ernie_key}
    ERNIE_SECRET_KEY={ernie_sec}
    HF_TOKEN={hf_token}
""")
ENV_FILE.write_text(content)
print(".keys.env written – you're ready to run benchmarks!")
