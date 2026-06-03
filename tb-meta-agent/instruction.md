# Terminal Bench Meta-Agent Optimization Task

## CRITICAL RESTRICTIONS

**YOU MAY ONLY USE THE PROVIDED API ENDPOINT**

You are **strictly prohibited** from:
- Using any other LLM APIs (Anthropic, Google, Cohere, etc.)
- Running local models (Ollama, llama.cpp, vLLM, etc.)
- Accessing pre-trained models from HuggingFace, ModelScope, etc.
- Downloading Terminal Bench datasets or solutions from the web.
- Using any AI services not explicitly provided.
- Directly calling external API endpoints — all model calls must go through the provided proxy.

**You may ONLY use:**
- The API endpoint provided via environment variables (`TASK_MODEL_API_BASE`, `TASK_MODEL_API_KEY`, `TASK_MODEL_NAME`)
- Your own logic and algorithms

**Your agent code must use `TASK_MODEL_API_BASE` and `TASK_MODEL_API_KEY` for all LLM calls.** These point to a local model proxy that enforces the assigned model. Do NOT use `OPENAI_*`, `ANTHROPIC_*`, or any other credentials — they are not available to your agent at runtime.

**Violation of these restrictions will result in immediate disqualification.**

Network monitoring is in place to detect unauthorized API usage.

---

## Objective

Create and iteratively improve a **Terminal Bench solving agent** by writing code in `/workspace`.

- Primary artifact: `/workspace/agent.py`
  - You may add helper modules

This task includes a local `POST /evaluate/agent` proxy (Flask). It forwards evaluation to an external Harbor CLI HTTP service running at `HARBOR_CLI_SERVICE_URL`.

## What You Should Build

Implement a Harbor-compatible agent that solves Terminal Bench tasks when executed by the Harbor evaluator.

- Base class: `harbor.agents.base.BaseAgent`
- Required: implement `run(instruction, environment, context)`
- Save your entrypoint to: `/workspace/agent.py`

Harbor BaseAgent interface reference in this environment:
- `/app/tools/harbor/agents/base.py`

Default agent import path:
- `HARBOR_AGENT_IMPORT_PATH=agent:TBAgent` means "import class `TBAgent` from `agent.py`" in the Harbor CLI working directory (`HARBOR_CLI_CWD`).

Minimal skeleton:

```python
from __future__ import annotations

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

class TBAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "tb-agent"

    def version(self) -> str | None:
        return "0.1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        # Initialize tools, write config files, etc.
        _ = environment
        return None

    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        # Use `environment.exec(...)` and other environment helpers to interact with the terminal.
        # Populate `context` with any metadata you want to persist.
        #
        # The Harbor evaluator will determine success by running Terminal Bench grading logic
        # against the resulting terminal state.
        _ = instruction, environment, context
        return None
```

## Development Evaluation (unified evaluate interface)

The evaluation proxy runs in a separate container and is reachable via the
`EVALUATION_API_URL` environment variable (pre-set in the container):

- `POST $EVALUATION_API_URL/evaluate/agent`

It calls an **external** Harbor CLI service (configured by env var):

Example:

You can pass `"first_k": N` to only evaluate the first N tasks — useful for fast iteration:

```python
import os, requests

eval_api = os.environ.get("EVALUATION_API_URL", "http://evaluation-api:8080")
resp = requests.post(f"{eval_api}/evaluate/agent", json={
    "agent_file": "/workspace/agent.py",
    "timeout": 21600,
    "first_k": 10  # only run the first 10 tasks
})
print(resp.status_code)
print(resp.json())
```

Notes:
- The proxy serializes evaluations. If one is already running, it returns HTTP `409` with error `"another eval is running"`. To stop the currently running evaluation, call:
  ```python
  requests.post(f"{eval_api}/evaluate/agent", json={"kill_running": True})
  ```
  You can also include `kill_running: true` in the same JSON body as `agent_file` to do a best-effort "kill then run" in a single request. Then retry your evaluation request if you still get `409`.
- The proxy calls `/v1/cli/exec` on the cli_service host to execute:
  - `harbor run -d <dataset> ... --agent-import-path <path>`
- The agent import path is configured via `HARBOR_AGENT_IMPORT_PATH` (default: `agent:TBAgent`).
- The working directory for Harbor CLI import is controlled by `HARBOR_CLI_CWD`.
- The proxy forwards model config into the Harbor-run subprocess environment:
  - `TASK_MODEL_API_BASE`, `TASK_MODEL_API_KEY`, `TASK_MODEL_NAME`

## Time Budget

The meta-agent development phase is configured for **24 hours**. The verifier phase (the final test run) is configured for another **24 hours**.

## Final Evaluation

At the end of the development phase, the verifier imports `/workspace/agent.py` (entrypoint `agent:TBAgent`) and scores it on a held-out test dataset. The entire `/workspace` directory is used as the working directory, so any helper modules sitting next to `agent.py` are picked up as well. Whatever is in `/workspace` at that moment is your final submission.
