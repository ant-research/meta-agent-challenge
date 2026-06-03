import os
from terminus_2 import Terminus2


class TBAgent(Terminus2):
    """Terminus-2 wrapper that reads TASK_MODEL_* env vars."""

    def __init__(self, **kwargs):
        api_key = os.environ.get("TASK_MODEL_API_KEY", "")
        api_base = os.environ.get("TASK_MODEL_API_BASE", "")
        if api_key:
            llm_kwargs = kwargs.get("llm_kwargs") or {}
            llm_kwargs["api_key"] = api_key
            kwargs["llm_kwargs"] = llm_kwargs
        if api_base:
            kwargs.setdefault("api_base", api_base)
        super().__init__(**kwargs)
