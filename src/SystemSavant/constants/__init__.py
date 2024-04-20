from pathlib import Path

CONFIG_PATH = Path("config/config.yaml")
PARAMS_PATH = Path("config/params.yaml")
PERSIST_DIR = "/db"

SYSTEM_PROMPT = """
[INST] <<SYS>>
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
<</SYS>>[/INST]
"""

QUERY_PROMPT = "{query_str}"

HF_TOKEN = "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"