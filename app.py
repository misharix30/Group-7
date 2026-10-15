"""Local Streamlit chat: only the latest turn exists in session memory."""
import logging
import os
from time import perf_counter

# Keep local questions out of tracing services, including inherited notebook settings.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import streamlit as st

from src.models import load_runtime
from src.ui import STYLES, example_questions, message_html

LOGGER = logging.getLogger(__name__)
st.set_page_config(page_title="اعرف دوائك", page_icon="💊", layout="centered")
st.markdown(STYLES, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def runtime():
    return load_runtime()


def submit(question=None):
    if st.session_state.get("busy", False):
        return
    st.session_state["request_started_at"] = perf_counter()
    text = question if question is not None else st.session_state.get("question_input", "")
    # Replace the previous turn before processing; never accumulate a history list.
    st.session_state["latest_turn"] = None
    st.session_state["pending_question"] = text
    st.session_state["busy"] = True


st.session_state.setdefault("busy", False)
st.session_state.setdefault("latest_turn", None)
st.session_state.setdefault("pending_question", None)
busy = st.session_state["busy"]

st.markdown('''<div class="brand"><span class="badge">مساعد التداخلات الدوائية</span>
<h1>اعرف دوائك</h1><p>اسأل عن التداخل بين دواءين، واحصل على شرح واضح يستند إلى البيانات المتاحة.</p>
</div>''', unsafe_allow_html=True)
st.markdown('<div class="section-label">جرّب سؤالًا</div>', unsafe_allow_html=True)
examples = example_questions()
if examples:
    for column, (label, question) in zip(st.columns(len(examples)), examples):
        column.button(label, on_click=submit, args=(question,), disabled=busy,
                      use_container_width=True)
st.markdown('<div class="privacy">تظهر أحدث إجابة فقط؛ تُستبدل عند إرسال سؤال جديد، ولا تُحفظ المحادثات.</div>',
            unsafe_allow_html=True)
chat_slot = st.empty()
chat_area = chat_slot.container()
st.chat_input("اكتب سؤالك عن دواءين…", key="question_input", disabled=busy, on_submit=submit)

pending = st.session_state["pending_question"]
if busy and pending is not None:
    chat_slot.empty()
    chat_area = chat_slot.container()
    if not pending.strip():
        st.session_state["latest_turn"] = {"question": "", "error": "اكتب سؤالًا عن التداخل بين دواءين أولًا."}
    else:
        with chat_area:
            with st.chat_message("user", avatar="👤"):
                st.markdown(message_html(pending), unsafe_allow_html=True)
            try:
                with st.status("تهيئة المساعد…", expanded=True) as state:
                    engine = runtime()
                    state.update(label="جاري تحليل سؤالك…")
                    answer = engine.run(pending)
                    state.update(label="اكتمل التحليل", state="complete", expanded=False)
                st.session_state["latest_turn"] = {"question": pending, "answer": answer}
            except Exception as error:
                # No file logger: technical details go to the terminal only.
                LOGGER.exception("Interaction request failed")
                st.session_state["latest_turn"] = {
                    "question": pending,
                    "error": getattr(error, "user_message", "تعذّر إكمال التحليل بسبب خطأ في التشغيل. تفاصيل المشكلة تظهر في نافذة التشغيل؛ راجعها لتحديد السبب قبل المحاولة مجددًا."),
                }
    if st.session_state.get("request_started_at") is not None:
        st.session_state["latest_turn"]["elapsed_seconds"] = (
            perf_counter() - st.session_state["request_started_at"]
        )
    st.session_state.pop("request_started_at", None)
    st.session_state["pending_question"] = None
    st.session_state["busy"] = False
    st.rerun()

turn = st.session_state["latest_turn"]
with chat_area:
    if turn is None:
        st.markdown('''<div class="welcome" dir="rtl"><div style="font-size:2.1rem">💬</div>
<h3>ما سؤالك عن دوائك؟</h3><p>اذكر اسمَي دوائين في سؤال واحد.<br>يمكنك السؤال بالعربية .</p></div>''',
                    unsafe_allow_html=True)
    else:
        if turn["question"]:
            with st.chat_message("user", avatar="👤"):
                st.markdown(message_html(turn["question"]), unsafe_allow_html=True)
        with st.chat_message("assistant", avatar="💊"):
            if "error" in turn:
                st.error(turn["error"])
            else:
                answer = turn["answer"]
                if answer.medgemma_used:
                    st.caption("شرح التداخل")
                elif answer.status == "out_of_scope":
                    st.caption("يحتاج السؤال إلى توضيح أو يقع خارج نطاق المساعد")
                else:
                    st.info("لم يتم التحقق من تداخل مطابق في السجل المسترجع. هذه النتيجة لا تعني أن الجمع بين الدواءين آمن.")
                st.markdown(message_html(answer.response), unsafe_allow_html=True)
                if turn.get("elapsed_seconds") is not None:
                    st.caption(f"مدة الاستجابة: {turn['elapsed_seconds']:.2f} ثانية")
                if answer.record is not None:
                    with st.expander("التفاعل الأصلي والمصدر"):
                        st.markdown(message_html(answer.record["interaction"]), unsafe_allow_html=True)
                        st.caption(f"المصدر: ddi_cleaned.csv · رقم السجل: {answer.record['record_id']}")
