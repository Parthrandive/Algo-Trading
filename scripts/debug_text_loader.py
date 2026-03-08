import logging
from src.agents.preprocessing.loader import TextLoader

logging.basicConfig(level=logging.INFO)

loader = TextLoader()
df = loader.load("db_virtual", "test")
print(f"Textual DataFrame shape: {df.shape}")
if not df.empty:
    print(df.head())
