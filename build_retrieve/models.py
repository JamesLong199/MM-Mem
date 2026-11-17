"""
Model management and inference utility module
- Base model: vLLM deployment (aligned with reference/VideoMME/run_videomme.py)
- Fine-tuned model: transformers loading (only for L2 decisions, logprobs not needed)
- BGE embedding / reranker models
- logprobs extraction: using vLLM native logprobs API
"""

import math
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image

from config import MemoryConfig

logger = logging.getLogger(__name__)


# ============================================================
# vLLM logprobs extraction (aligned with retrieval_utils.py)
# ============================================================

OPTION_LABELS = ("A", "B", "C", "D")
RELEVANCE_LABELS = ("yes", "no")


def normalize_answer(answer_str):
    """Extract option letter A/B/C/D"""
    if not isinstance(answer_str, str):
        return ""
    for char in answer_str.strip():
        if char.isalpha():
            return char.upper()
    return ""


def normalize_binary_label(text):
    """Normalize to yes/no"""
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", text.strip()).lower()
    return cleaned if cleaned in RELEVANCE_LABELS else ""


def aggregate_label_probabilities_vllm(logprobs_dict, labels, normalizer):
    """
    Extract normalized label probability distribution from vLLM logprob dict.
    Fully aligned with reference/VideoMME/retrieval_utils.py.
    """
    label_masses = {label: 0.0 for label in labels}
    if logprobs_dict is None:
        return label_masses

    for token_id, logprob_obj in logprobs_dict.items():
        token_text = logprob_obj.decoded_token
        logprob_value = logprob_obj.logprob
        if logprob_value <= -100:
            continue
        label = normalizer(token_text)
        if label not in label_masses:
            continue
        label_masses[label] += math.exp(logprob_value)

    total_mass = sum(label_masses.values())
    if total_mass <= 0:
        return {label: 0.0 for label in labels}
    return {label: label_masses[label] / total_mass for label in labels}


def extract_option_probs(vllm_output, labels=("A", "B", "C", "D")):
    """
    Extract option probability distribution from vLLM output.
    """
    logprobs_list = vllm_output.outputs[0].logprobs
    first_token_logprobs = logprobs_list[0] if logprobs_list else None
    return aggregate_label_probabilities_vllm(
        first_token_logprobs, labels, normalize_answer
    )


def extract_yes_no_probs(vllm_output):
    """
    Extract yes/no probabilities from vLLM output.
    """
    logprobs_list = vllm_output.outputs[0].logprobs
    first_token_logprobs = logprobs_list[0] if logprobs_list else None
    label_probs = aggregate_label_probabilities_vllm(
        first_token_logprobs, RELEVANCE_LABELS, normalize_binary_label
    )
    return {"yes": label_probs.get("yes", 0.0), "no": label_probs.get("no", 0.0)}


# ============================================================
# ModelManager
# ============================================================

class ModelManager:
    """
    Lazy-load and cache all models:
    - Base model: vLLM LLM (for caption/entity extraction/VQA/logprobs)
    - Fine-tuned model: transformers (only for L2 ADD_NEW/MERGE/DISCARD decisions)
    - BGE embedding / reranker
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._llm = None
        self._processor = None
        self._finetuned_model = None
        self._finetuned_processor = None
        self._embedding_model = None
        self._reranker_model = None

    def get_base_model(self):
        """
        Get vLLM LLM instance and processor.
        Returns (llm, processor).
        """
        if self._llm is None:
            try:
                from vllm import LLM
                from transformers import AutoProcessor
                logger.info(f"Initializing vLLM: {self.config.base_model_path}")
                self._llm = LLM(
                    model=self.config.base_model_path,
                    tensor_parallel_size=self.config.tensor_parallel_size,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    trust_remote_code=True,
                    max_model_len=self.config.max_model_len,
                    limit_mm_per_prompt={"image": self.config.max_images_per_prompt},
                    seed=self.config.seed,
                )
                self._processor = AutoProcessor.from_pretrained(self.config.base_model_path)
                logger.info("vLLM initialization complete")
            except Exception as exc:
                logger.warning("vLLM unavailable (%s); using transformers SDPA", exc)
                from transformers_backend import TransformersLLM
                self._llm = TransformersLLM(self.config.base_model_path)
                self._processor = self._llm.processor

        return self._llm, self._processor

    def get_finetuned_model(self):
        """
        Get fine-tuned model (vLLM deployment) for L2 decisions.
        Falls back to base model when no fine-tuned model is available.
        """
        if not self.config.finetuned_model_path:
            logger.warning("No fine-tuned model path specified, L2 decisions fall back to base model")
            return self.get_base_model()

        if self._finetuned_model is None:
            from vllm import LLM
            from transformers import AutoProcessor

            logger.info(f"Initializing vLLM fine-tuned model: {self.config.finetuned_model_path}")
            self._finetuned_model = LLM(
                model=self.config.finetuned_model_path,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=True,
                max_model_len=self.config.max_model_len,
                limit_mm_per_prompt={"image": self.config.max_images_per_prompt},
                seed=self.config.seed,
            )
            self._finetuned_processor = AutoProcessor.from_pretrained(
                self.config.finetuned_model_path
            )
            logger.info("vLLM fine-tuned model initialization complete")

        return self._finetuned_model, self._finetuned_processor

    def get_embedding_model(self):
        """Get BGE embedding model"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.config.embedding_model_name)
        return self._embedding_model

    def get_reranker_model(self):
        """Get BGE reranker model"""
        if self._reranker_model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {self.config.reranker_model_name}")
            self._reranker_model = CrossEncoder(self.config.reranker_model_name)
        return self._reranker_model


# ============================================================
# Message construction
# ============================================================

def build_messages(
    text: str,
    images=None,
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """
    Build messages in Qwen VL format.
    images supports two types:
      - List[str]: file path list (recommended, avoids vLLM cache issues)
      - List[PIL.Image]: PIL image objects
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    content = []
    if images:
        for img in images:
            if isinstance(img, str):
                # File path - aligned with reference, vLLM engine loads internally
                content.append({"type": "image", "image": img})
            else:
                # PIL Image object
                content.append({"type": "image", "image": img})
    content.append({"type": "text", "text": text})

    messages.append({"role": "user", "content": content})
    return messages


# ============================================================
# vLLM input preparation (aligned with run_videomme.py prepare_inputs_for_vllm)
# ============================================================


PATCH_FACTOR = 32
CONTEXT_LIMIT = 262144
CONTEXT_MARGIN = 2048
IMAGE_TOKEN_OVERHEAD = 48
MIN_IMAGE_TOKENS = 4


def _choose_hw(height: int, width: int, spatial_budget: int, factor: int = PATCH_FACTOR):
    """Largest factor-aligned size under the token budget whose aspect stays nearest the original."""
    native_h = max(factor, round(height / factor) * factor)
    native_w = max(factor, round(width / factor) * factor)
    native_tokens = (native_h // factor) * (native_w // factor)
    if native_tokens <= spatial_budget:
        return native_h, native_w
    target_aspect = width / height
    best = None
    for h in range(factor, native_h + 1, factor):
        for w in range(factor, native_w + 1, factor):
            tokens = (h // factor) * (w // factor)
            if tokens > spatial_budget or tokens < 1:
                continue
            aspect_error = abs((w / h) - target_aspect) / target_aspect
            # Closest aspect first, then the least downscaling.
            key = (aspect_error, -tokens)
            if best is None or key < best[0]:
                best = (key, h, w)
    if best is None:
        return factor, factor
    return best[1], best[2]


def fit_images_to_context(messages, processor, context_limit: int = CONTEXT_LIMIT, margin: int = CONTEXT_MARGIN):
    """Shrink every image in a request, keeping aspect ratio, so the prompt fits the context window.

    Frames that already fit are left at their native resolution. Each image keeps its own aspect
    ratio and receives the same token budget.
    """
    items = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for element in content:
            if isinstance(element, dict) and element.get("type") == "image":
                items.append(element)
    if not items:
        return messages

    text_messages = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            kept = [
                element
                for element in content
                if not (isinstance(element, dict) and element.get("type") == "image")
            ]
            text_messages.append({**message, "content": kept or [{"type": "text", "text": ""}]})
        else:
            text_messages.append(message)
    prompt = processor.apply_chat_template(text_messages, tokenize=False, add_generation_prompt=True)
    text_tokens = len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    sizes = []
    for element in items:
        image = element["image"]
        if isinstance(image, str):
            with Image.open(image) as im:
                sizes.append(im.size)
        else:
            sizes.append(image.size)

    native_h = max(PATCH_FACTOR, round(sizes[0][1] / PATCH_FACTOR) * PATCH_FACTOR)
    native_w = max(PATCH_FACTOR, round(sizes[0][0] / PATCH_FACTOR) * PATCH_FACTOR)
    native_tokens = text_tokens + len(items) * (
        (native_h // PATCH_FACTOR) * (native_w // PATCH_FACTOR) + IMAGE_TOKEN_OVERHEAD
    )
    if native_tokens + margin <= context_limit:
        return messages

    budget = context_limit - margin - text_tokens
    per_image = max(MIN_IMAGE_TOKENS + IMAGE_TOKEN_OVERHEAD, budget // len(items))
    spatial_budget = max(MIN_IMAGE_TOKENS, per_image - IMAGE_TOKEN_OVERHEAD)
    chosen = [_choose_hw(height, width, spatial_budget) for width, height in sizes]
    for element, (h_bar, w_bar) in zip(items, chosen):
        element["resized_height"] = h_bar
        element["resized_width"] = w_bar
        element.pop("max_pixels", None)

    h_bar, w_bar = chosen[0]
    logger.info(
        "Context fit: %d images, native %dx%d would be %d tokens; scaled to %dx%d with text=%d",
        len(items),
        native_w,
        native_h,
        native_tokens,
        w_bar,
        h_bar,
        text_tokens,
    )
    return messages


def prepare_vllm_input(messages, processor):
    """
    Prepare vLLM input, aligned with reference.
    Uses qwen_vl_utils.process_vision_info to process multimodal content.
    """
    from qwen_vl_utils import process_vision_info

    fit_images_to_context(messages, processor)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        image_patch_size=16,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    mm_kwargs = {}
    if video_kwargs is not None:
        mm_kwargs.update(video_kwargs)

    result = {"prompt": text, "multi_modal_data": mm_data}
    if mm_kwargs:
        result["mm_processor_kwargs"] = mm_kwargs
    return result


# ============================================================
# vLLM inference functions (base model)
# ============================================================

def generate_text(
    llm, processor, messages: List[Dict],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    presence_penalty: float = 0.0,
) -> str:
    """
    Generate text using vLLM. Used for captions, entity extraction, etc.
    """
    if hasattr(llm, "generate_messages"):
        return llm.generate_messages(messages, max_new_tokens=max_new_tokens, temperature=temperature).strip()
    from vllm import SamplingParams

    vllm_input = prepare_vllm_input(messages, processor)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_new_tokens,
        presence_penalty=presence_penalty,
        stop_token_ids=[],
    )

    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    return outputs[0].outputs[0].text.strip()


def generate_with_logprobs(
    llm, processor, messages: List[Dict],
    logprobs: int = 10,
) -> Any:
    """
    Generate 1 token using vLLM and return logprobs.
    Returns the raw vLLM output object for use by extract_option_probs / extract_yes_no_probs.
    """
    if hasattr(llm, "generate_with_logprobs"):
        return llm.generate_with_logprobs(messages, topk=logprobs)
    from vllm import SamplingParams

    vllm_input = prepare_vllm_input(messages, processor)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=logprobs,
        stop_token_ids=[],
    )

    outputs = llm.generate([vllm_input], sampling_params=sampling_params)
    return outputs[0]
