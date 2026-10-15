"""Manual offline diagnosis; print metrics only, never save generated answers."""
import hashlib
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import torch
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.generation import ROLE_MAP
from src.models import load_runtime
from src.prompts import MEDGEMMA_SYSTEM_PROMPT, MEDGEMMA_USER_PROMPT


def report(label, response):
    print(json.dumps({
        "case": label,
        "contains_arabic": any("\u0621" <= char <= "\u064a" for char in response),
        "words": len(response.split()),
        "has_english_preamble": response.startswith("Final Explanation:"),
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-language-instruction", action="store_true")
    options = parser.parse_args()
    engine = load_runtime(lambda message: print(message, flush=True))
    question = "ما هو التفاعل الدوائي بين Trioxsalen و Verteporfin؟\nأجب بالعربي"
    if options.compare_language_instruction:
        # An experiment only: leave production prompts and decoding unchanged.
        interaction = (
            "Trioxsalen may increase the photosensitizing activities of Verteporfin."
        )
        inputs = {"question": question, "interaction": interaction}
        suffix = (
            "\n\nلغة الإجابة المطلوبة: العربية.\n"
            "اشرح التداخل أعلاه بالعربية فقط، مع إبقاء أسماء الأدوية كما هي.\n"
            "ابدأ بالشرح العربي مباشرة دون عنوان أو مقدمة بالإنجليزية."
        )
        localized_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=MEDGEMMA_SYSTEM_PROMPT),
            ("human", MEDGEMMA_USER_PROMPT + suffix),
        ])
        generator = engine.medgemma_chain.steps[1].bound
        for sample in (False, True):
            chain = (
                localized_prompt
                | generator.bind(max_new_tokens=2000, do_sample=sample, show_thinking=False)
                | engine.medgemma_chain.steps[2]
            )
            torch.manual_seed(42)
            report(f"arabic_suffix_sample_{sample}", chain.invoke(inputs))
        return
    torch.manual_seed(42)
    answer = engine.run(question)
    assert answer.medgemma_used, "The question did not reach MedGemma."
    inputs = {"question": question, "interaction": answer.record["interaction"]}
    prompt_value = engine.medgemma_chain.steps[0].invoke(inputs)
    messages = [
        {"role": ROLE_MAP[message.type],
         "content": [{"type": "text", "text": message.content}]}
        for message in prompt_value.to_messages()
    ]
    generator = engine.medgemma_chain.steps[1].bound.func
    captured = dict(zip(generator.__code__.co_freevars,
                        (cell.cell_contents for cell in generator.__closure__)))
    processor = captured["medgemma_processor"]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    print(json.dumps({
        "question_present_in_model_template": question in rendered,
        "current_system_prompt_present": MEDGEMMA_SYSTEM_PROMPT in rendered,
        "prompt_sha256": hashlib.sha256(MEDGEMMA_SYSTEM_PROMPT.encode()).hexdigest(),
        "input_tokens": len(processor.tokenizer.encode(rendered)),
    }), flush=True)
    report("fresh_runtime_sample_seed_42", answer.response)
    # Compare the same matched record and same prompt with deterministic decoding.
    greedy = (
        engine.medgemma_chain.steps[0]
        | engine.medgemma_chain.steps[1].bound.bind(
            max_new_tokens=2000, do_sample=False, show_thinking=False,
        )
        | engine.medgemma_chain.steps[2]
    )
    report("same_prompt_greedy", greedy.invoke(inputs))
    torch.manual_seed(7)
    report("same_prompt_sample_seed_7", engine.medgemma_chain.invoke(inputs))


if __name__ == "__main__":
    main()
