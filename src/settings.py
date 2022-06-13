import os

from dotenv import load_dotenv

load_dotenv()


MODEL_NAME_OR_PATH = os.getenv("MODEL_NAME_OR_PATH", "microsoft/DialoGPT-small")
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "microsoft/DialoGPT-small")
CONFIG_NAME = os.getenv("CONFIG_NAME", "microsoft/DialoGPT-small")
