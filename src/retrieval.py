"""Open the saved collection without recreating the dataset or embeddings."""
import json
import os

from src.config import DB_PATH, EMBEDDING_MODEL, ROOT


def database_directory():
    # Windows HNSW rejects the non-ASCII absolute project path. Use the same
    # ASCII relative path as the notebook, anchored explicitly to ROOT so a
    # launch from another directory also works. No data is moved or copied.
    if os.name == "nt" and not str(DB_PATH).isascii():
        os.chdir(ROOT)
        return DB_PATH.relative_to(ROOT).as_posix()
    return str(DB_PATH)


def open_retriever():
    if not (DB_PATH / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Saved database not found: {DB_PATH}")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from chromadb.config import Settings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
        show_progress=False,
    )
    store = Chroma(collection_name="ddi_interactions", persist_directory=database_directory(),
                   embedding_function=embeddings, create_collection_if_not_exists=False,
                   client_settings=Settings(anonymized_telemetry=False))
    if store._collection.count() == 0:
        raise RuntimeError("The saved interaction collection is empty.")
    return store.as_retriever(search_type="similarity", search_kwargs={"k": 1})


def format_context(documents):
    records = [{field: doc.metadata[field] for field in
                ("record_id", "drug1_name", "drug2_name", "interaction")}
               for doc in documents]
    return json.dumps(records, ensure_ascii=False, indent=2)
