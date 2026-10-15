# Generation functions copied verbatim; scoped to cached model instances.
import torch

ROLE_MAP = {"system": "system", "human": "user", "ai": "assistant"}

def build_generators(qwen_pipeline, medgemma_pipe):
    medgemma_processor = medgemma_pipe.processor
    medgemma_model = medgemma_pipe.model
    qwen_tokenizer = qwen_pipeline.tokenizer

    def generate_qwen_response(prompt_value, max_new_tokens=256, do_sample=False, temperature=0.7, top_p=0.8):
        messages = [
            {
                "role": ROLE_MAP[message.type],
                "content": message.content
            }
            for message in prompt_value.to_messages()
        ]

        prompt_text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        settings = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None
        }

        if not do_sample:
            settings["top_k"] = None

        output = qwen_pipeline(
            prompt_text,
            return_full_text=False,
            **settings
        )

        response = output[0]["generated_text"].strip()

        if not response:
            raise RuntimeError("Qwen returned an empty answer.")

        return response

    def generate_medgemma_response(prompt_value, max_new_tokens=2000, do_sample=False, temperature=0.9, top_p=0.7, show_thinking=False):
        messages = [
            {
                "role": ROLE_MAP[message.type],
                "content": [{"type": "text", "text": message.content}]
            }
            for message in prompt_value.to_messages()
        ]

        inputs = medgemma_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(medgemma_model.device)

        input_length = inputs["input_ids"].shape[-1]

        settings = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None
        }

        if not do_sample:
            settings["top_k"] = None

        with torch.inference_mode():
            output = medgemma_model.generate(**inputs, **settings)

        response = medgemma_processor.decode(
            output[0][input_length:],
            skip_special_tokens=True
        ).strip()

        if not response:
            raise RuntimeError("The model returned an empty answer.")

        if show_thinking:
            return response

        # Extract the final answer when thinking is hidden.
        if "<unused95>" in response:
            response = response.split("<unused95>", 1)[1].strip()
        elif "<unused94>thought" in response:
            raise RuntimeError(
                "Generation ended before a final answer was produced."
            )

        if not response:
            raise RuntimeError("No final answer was produced.")

        return response

    return generate_qwen_response, generate_medgemma_response
