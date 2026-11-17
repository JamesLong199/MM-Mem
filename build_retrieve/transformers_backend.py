"""Transformers stand-in for the vLLM calls in MM-Mem. vLLM has no aarch64 wheel."""
from __future__ import annotations

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# The cuDNN SDPA backend builds no plan on this B200 and then aborts.
torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


class _Logprob:
    def __init__(self, token: str, logprob: float) -> None:
        self.decoded_token = token
        self.logprob = logprob


class _Sequence:
    def __init__(self, text: str, logprobs) -> None:
        self.text = text
        self.logprobs = logprobs


class _RequestOutput:
    def __init__(self, text: str, logprobs) -> None:
        self.outputs = [_Sequence(text, logprobs)]


class TransformersLLM:
    def __init__(self, model_path: str) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        self.model.eval()

    def _inputs(self, messages):
        from qwen_vl_utils import process_vision_info
        from models import fit_images_to_context

        fit_images_to_context(messages, self.processor)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
        batch = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        return {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }

    def generate_messages(self, messages, max_new_tokens: int = 512, temperature: float = 0.0) -> str:
        inputs = self._inputs(messages)
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = max(temperature, 1e-5)
        with torch.no_grad():
            generated = self.model.generate(**inputs, **kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        text = self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0]
        return text

    def generate_with_logprobs(self, messages, topk: int = 10):
        inputs = self._inputs(messages)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        token_ids = output.sequences[:, prompt_len:]
        text = self.processor.batch_decode(token_ids, skip_special_tokens=True)[0]
        scores = output.scores[0][0].float()
        log_probs = torch.log_softmax(scores, dim=-1)
        values, indices = torch.topk(log_probs, k=topk)
        table = {}
        for index, value in zip(indices.tolist(), values.tolist()):
            token = self.processor.tokenizer.decode([index])
            table[index] = _Logprob(token, float(value))
        return _RequestOutput(text, [table])
