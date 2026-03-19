"""Agentic Deep Research System for the DeepScholar benchmark."""

def __getattr__(name: str):
    if name == "run_research":
        from .graph import run_research
        return run_research
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["run_research"]
