"""Open the actual saved index from an unrelated launch directory, without models."""
import os
import subprocess
import sys

from src.config import ROOT


def test_existing_collection_opens_from_another_directory(tmp_path):
    code = (
        "import sys; sys.path.insert(0, " + repr(str(ROOT)) + "); "
        "import chromadb; from chromadb.config import Settings; "
        "from src.retrieval import database_directory; "
        "client=chromadb.PersistentClient(path=database_directory(), "
        "settings=Settings(anonymized_telemetry=False)); "
        "collection=client.get_collection('ddi_interactions', embedding_function=None); "
        "print(collection.count())"
    )
    environment = dict(os.environ, PYTHONUTF8="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0
