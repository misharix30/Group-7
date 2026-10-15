"""Run with python -m src.check_environment; does not load LLM weights."""
import importlib
import importlib.metadata
import json
import platform
import sqlite3

from src.config import DB_PATH, ROOT

PACKAGES = {
    "torch": "torch", "torchvision": "torchvision", "transformers": "transformers",
    "accelerate": "accelerate", "bitsandbytes": "bitsandbytes",
    "sentence-transformers": "sentence_transformers", "langchain-core": "langchain_core",
    "langchain-huggingface": "langchain_huggingface", "langchain-chroma": "langchain_chroma",
    "chromadb": "chromadb", "streamlit": "streamlit", "pandas": "pandas",
    "Pillow": "PIL", "ipykernel": "ipykernel", "jupyterlab": "jupyterlab", "pytest": "pytest",
}


def inspect_environment():
    report = {"python": platform.python_version(), "packages": {}, "errors": []}
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            report["packages"][package] = importlib.metadata.version(package)
        except Exception as error:
            report["errors"].append(f"{package}: {type(error).__name__}: {error}")
    import torch
    report["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        report["cuda"].update(gpu=torch.cuda.get_device_name(0),
                              free_gib=round(free / 1024**3, 2),
                              total_gib=round(total / 1024**3, 2),
                              capability=list(torch.cuda.get_device_capability(0)))
        # Exercise a real CUDA kernel and BF16, without loading models.
        sample = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        report["cuda"]["bf16_matmul_ok"] = bool(torch.all(sample @ sample == 16).item())
        del sample
        # Exercise the same NF4 backend used by the notebook.
        from bitsandbytes.functional import quantize_4bit, dequantize_4bit
        sample = torch.randn(256, device="cuda", dtype=torch.bfloat16)
        quantized, state = quantize_4bit(sample, quant_type="nf4")
        restored = dequantize_4bit(quantized, state)
        report["cuda"]["nf4_ok"] = restored.shape == sample.shape and bool(torch.isfinite(restored).all())
        del sample, quantized, state, restored
    db = DB_PATH / "chroma.sqlite3"
    if not db.is_file():
        report["errors"].append("Saved Chroma database is missing.")
    else:
        # SQLite read-only access avoids any migrations or modifications.
        with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT name FROM collections").fetchall()
            report["database"] = {"path": str(db.relative_to(ROOT)), "collections": [row[0] for row in rows]}
            if "ddi_interactions" not in report["database"]["collections"]:
                report["errors"].append("ddi_interactions collection is missing.")
    return report


if __name__ == "__main__":
    result = inspect_environment()
    output = ROOT / "results" / "environment_check.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(1 if result["errors"] else 0)
