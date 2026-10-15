"""Streamlit event tests: latest-turn replacement, validation and escaped output."""
from unittest.mock import Mock, patch
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest
import streamlit as st

from src.config import ROOT
from src.pipeline import Answer
from src.pipeline import InteractionPipeline, validate_output
from src.output_parser import parse_qwen_output
from src.ui import message_html
from langchain_core.runnables import RunnableLambda


def test_chat_replaces_previous_turn_without_history():
    fake = Mock()
    fake.run.side_effect = [Answer("valid", "FIRST_ANSWER", True), Answer("valid", "SECOND_ANSWER", True)]
    st.cache_resource.clear()
    with patch("src.models.load_runtime", return_value=fake):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        assert not app.exception
        app.chat_input[0].set_value("FIRST_QUESTION").run()
        assert not app.exception
        assert "FIRST_ANSWER" in " ".join(item.value for item in app.markdown)
        assert app.session_state["latest_turn"]["elapsed_seconds"] >= 0
        assert sum("مدة الاستجابة:" in item.value for item in app.caption) == 1
        assert "request_started_at" not in app.session_state
        app.chat_input[0].set_value("SECOND_QUESTION").run()
        assert not app.exception
        rendered = " ".join(item.value for item in app.markdown)
        assert "SECOND_ANSWER" in rendered and "FIRST_ANSWER" not in rendered
        assert "FIRST_QUESTION" not in rendered
        assert not app.chat_input[0].disabled
        assert fake.run.call_count == 2
        assert app.session_state["latest_turn"]["question"] == "SECOND_QUESTION"
        elapsed = app.session_state["latest_turn"]["elapsed_seconds"]
        assert elapsed >= 0
        assert sum(
            item.value == f"مدة الاستجابة: {elapsed:.2f} ثانية" for item in app.caption
        ) == 1
        assert "request_started_at" not in app.session_state
        assert "history" not in app.session_state
    st.cache_resource.clear()


def test_blank_question_never_loads_models():
    st.cache_resource.clear()
    with patch("src.models.load_runtime") as load:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.chat_input[0].set_value("   ").run()
        assert not app.exception
        assert app.error
        load.assert_not_called()


def test_response_html_is_escaped_and_direction_is_correct():
    assert '&lt;script&gt;' in message_html('<script>alert(1)</script>')
    assert 'dir="rtl"' in message_html("هذا شرح التداخل مع Drug A")
    assert 'dir="ltr"' in message_html("Drug A interacts with Drug B")


def test_unescaped_qwen_quotes_render_as_a_message_instead_of_an_error():
    raw = '{"status":"out_of_scope","response":"هذا السؤال يتضمن كلمة "مثال" خارج النطاق."}'
    retriever = Mock()
    retriever.invoke.return_value = [SimpleNamespace(metadata={
        "record_id": 1, "drug1_name": "Drug A", "drug2_name": "Drug B", "interaction": "Interaction.",
    })]
    generate = Mock(return_value=raw)
    chain = RunnableLambda(generate) | RunnableLambda(parse_qwen_output) | RunnableLambda(validate_output)
    medgemma = Mock()
    engine = InteractionPipeline(retriever, chain, medgemma)
    st.cache_resource.clear()
    with patch("src.models.load_runtime", return_value=engine):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        app.chat_input[0].set_value("سؤال خارج نطاق التداخلات").run()
        assert not app.exception and not app.error
        rendered = " ".join(item.value for item in app.markdown)
        assert 'كلمة &quot;مثال&quot;' in rendered
        assert generate.call_count == 1
        assert retriever.invoke.call_count == 1
        medgemma.invoke.assert_not_called()
    st.cache_resource.clear()
