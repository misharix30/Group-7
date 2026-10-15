"""Presentation helpers; model responses are escaped before HTML rendering."""
import csv
import html
import re

from src.config import CSV_PATH

STYLES = """
<style>
.stApp { background: radial-gradient(ellipse at top right, #e5f3f3 0%, #f5f8fa 55%); }
.block-container { max-width: 920px; padding-top: 2.8rem; padding-bottom: 7rem; }
header[data-testid="stHeader"] { background: transparent; }
.brand { direction: rtl; text-align: right; }
.brand .badge { display: inline-block; color: #087e8b; background: #e0f0ef;
    border-radius: 30px; padding: .35rem .8rem; font-size: .85rem; font-weight: 600; }
.brand h1 { color: #163341; font-size: clamp(2rem, 5vw, 3rem); padding: .6rem 0 .2rem; }
.brand p { color: #526a78; font-size: 1.08rem; line-height: 1.8; margin-bottom: 1.2rem; }
.welcome { text-align: center; background: rgba(255,255,255,.82); border: 1px solid #dce8ec;
    border-radius: 24px; padding: 2.6rem 1.4rem; margin-top: 1rem; color: #526a78; }
.welcome h3 { color: #163341; margin: .3rem 0; }
.message { white-space: pre-wrap; line-height: 1.95; font-size: 1.05rem;
    overflow-wrap: anywhere; unicode-bidi: plaintext; }
.message[dir="rtl"] { text-align: right; }
.message[dir="ltr"] { text-align: left; }
[data-testid="stChatMessage"] { border-radius: 18px; border: 1px solid #e0e9ed; padding: 1.2rem; }
[data-testid="stChatInput"] { border-radius: 18px; border: 1px solid #c6dce0; }
[data-testid="stBottom"] > div { background: #f5f8fa; }
[data-testid="stButton"] button { border-radius: 12px; border-color: #d6e5e9; }
[data-testid="stExpander"] { border-radius: 14px; }
.section-label { direction: rtl; text-align: right; font-weight: 600; color: #526a78; }
.privacy { direction: rtl; text-align: right; font-size: .8rem; color: #758892; margin: .5rem 0; }
@media(max-width: 640px) { .block-container { padding: 1.5rem 1rem 6rem; }
    .welcome { padding: 1.8rem 1rem; } [data-testid="stChatMessage"] { padding: .8rem; } }
</style>
"""


def message_html(text):
    # Use the first strong letter to select direction; mixed drug names remain readable.
    direction = "ltr"
    for character in text:
        if re.match(r"[\u0600-\u06ff]", character):
            direction = "rtl"
            break
        if character.isalpha():
            break
    return f'<div class="message" dir="{direction}">{html.escape(text)}</div>'


def example_questions():
    # Read just three saved rows; do not clean or transform the source file.
    if not CSV_PATH.is_file():
        return []
    examples = []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            first, second = row.get("Drug1_Name"), row.get("Drug2_Name")
            if first and second:
                question = f"ما هو التفاعل الدوائي بين {first} و {second}؟"
                examples.append((f"{first} + {second}", question))
            if len(examples) == 3:
                break
    return examples
