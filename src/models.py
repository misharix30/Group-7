"""Load the original quantized pipelines and build their unchanged prompts."""
from src.config import DB_PATH, MEDGEMMA_MODEL, QWEN_MODEL
from src.pipeline import GPU_LOCK, InteractionPipeline, validate_output


class SetupError(RuntimeError):
    def __init__(self, detail, user_message):
        super().__init__(detail)
        self.user_message = user_message


def load_runtime(progress=lambda message: None):
    # Imports and weights are lazy: the interface can open before GPU setup.
    import torch
    from transformers import BitsAndBytesConfig, pipeline
    from transformers.utils import logging
    from langchain_core.messages import SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from src.output_parser import parse_qwen_output
    from src.generation import build_generators
    from src.prompts import (VERIFY_SYSTEM_PROMPT, MEDGEMMA_SYSTEM_PROMPT,
                            VERIFY_USER_PROMPT, MEDGEMMA_USER_PROMPT)
    from src.retrieval import open_retriever

    with GPU_LOCK:
        if not (DB_PATH / "chroma.sqlite3").is_file():
            raise SetupError(f"Saved database not found: {DB_PATH}",
                             "ملفات بيانات التداخلات غير متوفرة في موقعها المطلوب. راجع دليل تشغيل المشروع.")
        if not torch.cuda.is_available():
            raise SetupError("CUDA is unavailable; enable a compatible NVIDIA runtime.",
                             "تعذّر الوصول إلى بطاقة الرسوميات المطلوبة لتشغيل المساعد. راجع إعدادات البيئة في دليل التشغيل.")
        free, _ = torch.cuda.mem_get_info()
        if free < 6 * 1024**3:
            raise SetupError("GPU memory is occupied. Close the notebook kernel before loading models.",
                             "الذاكرة المتاحة على بطاقة الرسوميات غير كافية للتحميل. أغلق جلسة النوتبوك التي حمّلت النماذج، ثم حاول مجددًا.")
        logging.set_verbosity_error()
        progress("تحميل نموذج تحليل السؤال…")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
        )
        qwen = pipeline(task="text-generation", model=QWEN_MODEL,
                        device_map="auto", dtype=torch.bfloat16,
                        model_kwargs={"quantization_config": quantization_config})
        progress("تحميل نموذج شرح التداخل…")
        medgemma = pipeline(task="image-text-to-text", model=MEDGEMMA_MODEL,
                            device_map="auto", dtype=torch.bfloat16,
                            model_kwargs={"quantization_config": quantization_config})
        generate_qwen, generate_medgemma = build_generators(qwen, medgemma)
        progress("فتح بيانات التداخلات المحفوظة…")
        retriever = open_retriever()
        verify_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=VERIFY_SYSTEM_PROMPT), ("human", VERIFY_USER_PROMPT),
        ])
        medgemma_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=MEDGEMMA_SYSTEM_PROMPT), ("human", MEDGEMMA_USER_PROMPT),
        ])
        qwen_chain = (verify_prompt
                      | RunnableLambda(generate_qwen).bind(max_new_tokens=256, do_sample=False)
                      | RunnableLambda(parse_qwen_output) | RunnableLambda(validate_output))
        medgemma_chain = (medgemma_prompt
                         | RunnableLambda(generate_medgemma).bind(
                             max_new_tokens=2000, do_sample=True, show_thinking=False)
                         | StrOutputParser())
        return InteractionPipeline(retriever, qwen_chain, medgemma_chain)
