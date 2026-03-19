"""LLM client with pluggable backends (OpenAI API or local HF/LoRA)."""
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from deep_research_agent.config import get_config


_hf_pipeline = None  # lazy-loaded HF text-generation pipeline


def _get_openai_client(
    temperature: Optional[float],
    max_tokens: Optional[int],
    model: Optional[str],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatOpenAI:
    """Helper to create the ChatOpenAI client with correct endpoint mapping."""
    cfg = get_config().llm
    kwargs: dict[str, Any] = {
        "model": model or cfg.model,
        "temperature": temperature if temperature is not None else cfg.temperature,
        "max_tokens": max_tokens or cfg.max_tokens,
    }
    
    # Use provided arguments or fallback to config
    final_api_key = api_key or cfg.api_key
    final_base_url = base_url or cfg.base_url
    
    if final_api_key:
        kwargs["api_key"] = final_api_key
    if final_base_url:
        kwargs["base_url"] = final_base_url
        
    return ChatOpenAI(**kwargs)


def _get_hf_pipeline() -> Any:
    """Return a local HF text-generation pipeline (optionally with LoRA)."""
    global _hf_pipeline
    if _hf_pipeline is not None:
        return _hf_pipeline

    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    cfg = get_config().llm
    if not cfg.hf_model_name_or_path:
        raise RuntimeError(
            "LLM backend is set to 'hf_local' but hf_model_name_or_path is not configured."
        )

    model_name = cfg.hf_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype="auto")

    if cfg.hf_lora_path:
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, cfg.hf_lora_path)
            model = model.merge_and_unload()
        except ImportError as exc:
            raise RuntimeError("peft is required for LoRA adapters.") from exc

    _hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return _hf_pipeline


def invoke(
    prompt: str,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Single LLM call. Returns content string."""
    cfg = get_config().llm

    # 1. OpenAI / Groq / OpenAI-compatible backend
    if cfg.backend == "openai":
        # Signature now matches the call below
        llm = _get_openai_client(
            temperature=temperature if temperature is not None else cfg.temperature,
            max_tokens=max_tokens or cfg.max_tokens,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.api_key
        )
        
        messages: list[BaseMessage] = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        
        out = llm.invoke(messages)
        return out.content if hasattr(out, "content") else str(out)

    # 2. Local HF Fallback
    if cfg.backend == "hf_local":
        pipe = _get_hf_pipeline()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or cfg.max_tokens,
            "temperature": max(0.01, temperature if temperature is not None else cfg.temperature),
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
        }
        outputs = pipe(full_prompt, **gen_kwargs)
        text = outputs[0]["generated_text"]
        if text.startswith(full_prompt):
            return text[len(full_prompt):].strip()
        return text.strip()

    raise RuntimeError(f"Unknown LLM backend: {cfg.backend!r}")

class LLMWrapper:
    """Simple wrapper so other modules can call llm.invoke()."""

    def invoke(self, prompt: str, system: Optional[str] = None) -> str:
        return invoke(prompt=prompt, system=system)


def get_llm() -> LLMWrapper:
    """Return an object with an invoke() method."""
    return LLMWrapper()