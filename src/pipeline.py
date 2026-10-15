"""One retrieval and one Qwen call; MedGemma only on exact interaction match."""
from dataclasses import dataclass
from threading import RLock

from src.retrieval import format_context

# Share this lock across resource loading and every session using the GPU.
GPU_LOCK = RLock()


def validate_output(output):
    if (not isinstance(output, dict) or set(output) != {"status", "response"}
            or output["status"] not in {"valid", "out_of_scope"}
            or not isinstance(output["response"], str)):
        raise ValueError("The model returned an invalid output schema.")
    return output


@dataclass(frozen=True)
class Answer:
    status: str
    response: str
    medgemma_used: bool
    record: dict | None = None


class InteractionPipeline:
    def __init__(self, retriever, qwen_chain, medgemma_chain):
        self.retriever = retriever
        self.qwen_chain = qwen_chain
        self.medgemma_chain = medgemma_chain

    def run(self, question):
        with GPU_LOCK:
            documents = self.retriever.invoke(question)
            result = validate_output(self.qwen_chain.invoke({
                "question": question, "context": format_context(documents),
            }))
            matched = (result["status"] == "valid" and bool(documents)
                       and result["response"] == documents[0].metadata["interaction"])
            if not matched:
                return Answer(result["status"], result["response"], False)
            explanation = self.medgemma_chain.invoke({
                "question": question, "interaction": result["response"],
            })
            record = {key: documents[0].metadata[key] for key in
                      ("record_id", "drug1_name", "drug2_name", "interaction")}
            return Answer("valid", explanation, True, record)
