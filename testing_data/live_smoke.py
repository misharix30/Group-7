"""Manual real-model smoke checks. Saves metrics only, never answer text."""
import argparse
import csv
import json
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true", help="Use already cached weights only")
options = parser.parse_args()
if options.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from src.config import CSV_PATH, ROOT
from src.models import load_runtime


def main():
    started = time.monotonic()
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    first, second = row["Drug1_Name"], row["Drug2_Name"]
    cases = [
        ("greeting_ar", "مرحبا", "out_of_scope", False),
        ("valid_en", f"What is the interaction between {first} and {second}?", "valid", True),
        ("valid_ar", f"ما هو التفاعل الدوائي بين {first} و {second}؟", "valid", True),
        ("one_drug", f"Does {first} interact with other medications?", "out_of_scope", False),
        ("three_drugs", f"Can I take {first}, {second}, and Ibuprofen together?", "out_of_scope", False),
        ("valid_no_record", "What is the interaction between Tirzepatide and Lecanemab?", "valid", False),
    ]
    engine = load_runtime(lambda message: print(message, flush=True))
    print("Runtime loaded successfully.", flush=True)
    results = []
    for name, question, expected_status, expected_explanation in cases:
        answer = engine.run(question)
        correct = answer.status == expected_status and answer.medgemma_used == expected_explanation
        item = {"case": name, "status": answer.status, "medgemma_used": answer.medgemma_used,
                "nonempty_response": bool(answer.response.strip()), "passed": correct}
        if answer.medgemma_used:
            item["response_words"] = len(answer.response.split())
        results.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    report = {"offline": options.offline, "elapsed_seconds": round(time.monotonic() - started, 2),
              "cases": results, "all_passed": all(item["passed"] for item in results)}
    (ROOT / "results" / "live_smoke.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
