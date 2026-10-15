"""Project-relative paths and unchanged notebook model identifiers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "ddi_chroma_db_backup" / "ddi_chroma_db"
CSV_PATH = ROOT / "ddi_cleaned.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "Prompt_Engineering_with_RAG_Organized.ipynb"
QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MEDGEMMA_MODEL = "google/medgemma-1.5-4b-it"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
