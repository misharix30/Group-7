"""Transfer regression checks using controlled model outputs, not LLM accuracy tests."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.pipeline import InteractionPipeline

RECORD = {"record_id": 7, "drug1_name": "Drug A", "drug2_name": "Drug B",
          "interaction": "Drug A may increase the effects of Drug B."}


def engine(output, documents=None):
    retriever = Mock()
    retriever.invoke.return_value = [SimpleNamespace(metadata=RECORD.copy())] if documents is None else documents
    qwen = Mock()
    qwen.invoke.return_value = output
    medgemma = Mock()
    medgemma.invoke.return_value = "Explanation in the requested language."
    return InteractionPipeline(retriever, qwen, medgemma)


@pytest.mark.parametrize("question", [
    "ما هو التفاعل بين Drug A و Drug B؟",
    "What is the interaction between Drug A and Drug B?",
    "What is the interaction between Drug B and Drug A?",
])
def test_current_question_and_exact_record_reach_medgemma(question):
    app = engine({"status": "valid", "response": RECORD["interaction"]})
    answer = app.run(question)
    assert answer.medgemma_used
    app.retriever.invoke.assert_called_once_with(question)
    app.qwen_chain.invoke.assert_called_once()
    qwen_input = app.qwen_chain.invoke.call_args.args[0]
    assert qwen_input["question"] == question
    assert json.loads(qwen_input["context"]) == [RECORD]
    app.medgemma_chain.invoke.assert_called_once_with({"question": question, "interaction": RECORD["interaction"]})
    assert answer.record == RECORD


@pytest.mark.parametrize("question", ["I love you", "Can I take Drug A?", "Can I take Drug A, Drug B and Drug C?"])
def test_out_of_scope_never_calls_medgemma(question):
    app = engine({"status": "out_of_scope", "response": "Please name exactly two medications."})
    answer = app.run(question)
    assert answer.response == "Please name exactly two medications."
    assert not answer.medgemma_used and answer.record is None
    app.medgemma_chain.invoke.assert_not_called()


@pytest.mark.parametrize("response", [
    "There is no matching record.", "لا يوجد سجل مطابق لهذين الدواءين.",
    RECORD["interaction"] + " ", RECORD["interaction"].lower(),
])
def test_valid_without_exact_match_does_not_call_medgemma(response):
    app = engine({"status": "valid", "response": response})
    answer = app.run("Drug A and Drug B")
    assert answer.response == response and not answer.medgemma_used
    app.retriever.invoke.assert_called_once()
    app.medgemma_chain.invoke.assert_not_called()


def test_empty_retrieval_cannot_trigger_explanation():
    app = engine({"status": "valid", "response": RECORD["interaction"]}, documents=[])
    assert not app.run("Drug A and Drug B").medgemma_used
    app.medgemma_chain.invoke.assert_not_called()


@pytest.mark.parametrize("output", [
    {"status": "valid", "response": ["text"]},
    {"status": "unknown", "response": "text"},
    {"status": "valid", "response": "text", "extra": True},
])
def test_invalid_schema_does_not_reach_medgemma(output):
    app = engine(output)
    with pytest.raises(ValueError):
        app.run("Drug A and Drug B")
    app.medgemma_chain.invoke.assert_not_called()
