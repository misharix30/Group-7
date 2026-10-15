"""Regression coverage for quotes in responses, without another model call."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda

from src.output_parser import parse_qwen_output
from src.pipeline import InteractionPipeline, validate_output


def test_arabic_inner_quotes_are_preserved():
    raw = '{"status": "out_of_scope", "response": "السؤال خارج النطاق؛ كلمة "إحراز" ليست اسم دواء."}'
    result = validate_output(parse_qwen_output(raw))
    assert result == {"status": "out_of_scope", "response": 'السؤال خارج النطاق؛ كلمة "إحراز" ليست اسم دواء.'}


@pytest.mark.parametrize("reverse", [False, True])
def test_fallback_preserves_escaped_and_unescaped_quotes_and_backslashes(reverse):
    body = r'Already \"escaped\", then "bare", path C:\\cache, newline\nnext.'
    if reverse:
        raw = '{"response":"' + body + '","status":"valid"}'
    else:
        raw = '{"status":"valid","response":"' + body + '"}'
    assert parse_qwen_output(raw)["response"] == 'Already "escaped", then "bare", path C:\\cache, newline\nnext.'


def test_valid_json_is_unchanged():
    expected = {"status": "valid", "response": 'Drug A causes "sedation".\nDo not rewrite.'}
    assert parse_qwen_output(json.dumps(expected)) == expected


@pytest.mark.parametrize("raw", [
    '{"status":"other","response":"a "quote" here"}',
    '{"status":"valid","response":"a "quote" here", "extra":1}',
    '{"status":"valid","response":"a "quote" here", "extra":"text"}',
    '{"status":"valid","response":"a "quote" here"} {"status":"valid","response":"second"}',
    r'{"status":"valid","response":"a "quote" with invalid \x escape"}',
    '{"status":"valid","response":["text"]}',
])
def test_does_not_invent_status_drop_extra_fields_or_accept_lists(raw):
    with pytest.raises((OutputParserException, ValueError)):
        validate_output(parse_qwen_output(raw))


@pytest.mark.parametrize("matched", [False, True])
def test_repaired_output_keeps_one_qwen_call_and_exact_match_gate(matched):
    response = 'Drug A increases "sedation" caused by Drug B.'
    status = "valid" if matched else "out_of_scope"
    generate = Mock(return_value='{"status":"' + status + '","response":"' + response + '"}')
    chain = RunnableLambda(generate) | RunnableLambda(parse_qwen_output) | RunnableLambda(validate_output)
    retriever = Mock()
    retriever.invoke.return_value = [SimpleNamespace(metadata={
        "record_id": 1, "drug1_name": "Drug A", "drug2_name": "Drug B", "interaction": response,
    })]
    medgemma = Mock()
    medgemma.invoke.return_value = "Explanation."
    engine = InteractionPipeline(retriever, chain, medgemma)
    answer = engine.run("Drug A and Drug B")
    assert generate.call_count == 1
    assert retriever.invoke.call_count == 1
    assert answer.medgemma_used == matched
    if matched:
        medgemma.invoke.assert_called_once_with({"question": "Drug A and Drug B", "interaction": response})
    else:
        medgemma.invoke.assert_not_called()
        assert answer.response == response
