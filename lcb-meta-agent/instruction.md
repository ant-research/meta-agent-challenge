# LiveCodeBench Meta-Agent Optimization Task

## ⚠️ CRITICAL RESTRICTIONS ⚠️

You may only use the OpenAI-compatible API endpoint provided in the environment.

Strictly prohibited:
- Using other model APIs (Anthropic / Google / Cohere / Together, etc.)
- Using local models (Ollama / vLLM / llama.cpp, etc.)
- Using external pretrained model services such as HuggingFace / ModelScope
- Downloading / accessing LiveCodeBench test data from external sources
- Hardcoding any API URL in `agent.py` or `/workspace` code (including `localhost`, `127.0.0.1`, or any third-party domain)
- Modifying the system file `/app/tools/openai_helper.py`
- Modifying the system file `/app/tools/base_agent.py`
- Creating a file named `openai_helper.py` or `base_agent.py` under `/workspace` (to avoid import conflicts)

Allowed:
- Using `TASK_MODEL_API_BASE` + `TASK_MODEL_API_KEY` + `TASK_MODEL_NAME`
- Building your agent workflow and code inside `/workspace`
- If you need to extend helper logic, create a file with a unique name (e.g. `/workspace/openai_helper_v2.py`) and explicitly import it from `agent.py`
- `BaseLCBAgent` must be imported from the system `base_agent`; replacing the base class implementation is not allowed
- `TASK_MODEL_API_BASE` must be read from environment variables only; if missing, exit immediately with an error — setting any default URL is not allowed
- The OpenAI client `base_url` must use `TASK_MODEL_API_BASE` directly; fallback URLs are not permitted

Mandatory output rules (must be satisfied):
- If the problem provides a non-empty `starter_code`, the generated code must begin with that `starter_code`, preserve its function signature / class structure, and complete the implementation on top of it
- If the problem does not provide a `starter_code`, the generated code must be a complete, executable stdin/stdout program
- When constructing the prompt for the OpenAI model, you must explicitly include the two output rules above — they must not be omitted or weakened
- Only one final answer per problem is allowed: `pred` must be a string

Violations of the above restrictions will result in an invalid submission; the environment performs network/API usage monitoring (including static code scanning).

---

## Objective

Your goal is to maximize code generation performance on LiveCodeBench, with a focus on improving:
- `accuracy` (problem pass rate)

You need to implement an agent that reads problems and outputs executable Python code; the evaluation backend will judge using hidden test cases.

## Task Description

You are free to design and optimize your code generation strategy, for example:
- Prompt template strategies (problem-type routing, platform routing, difficulty routing)
- Internal multi-candidate sampling and re-ranking strategies (output only 1 final answer)
- Failure retry and self-repair loops (you can adopt role-based division of labor to improve self-repair, e.g. let the OpenAI model act as an "error inspector" to review candidate results, or design a multi-stage workflow; for complex pipelines, set smaller `max_completion_tokens` per role stage to control latency and total runtime)
- Code post-processing (remove markdown fences, extract code blocks, format cleanup)
- Dual-channel strategy for function-call and stdin/stdout problems

---

## Environment Setup

### Available Tools

1. **OpenAI API Helper** (`/app/tools/openai_helper.py`)
   - Provides an OpenAI-compatible API call wrapper

2. **Model README** (`/workspace/model_readme.md`)
   - May contain information about the current model's capabilities, recommended parameters, context length, etc.
   - Recommended to read before designing your prompt/sampling strategy

3. **Evaluation API** (started automatically)
   - Address: `http://evaluation-api:8080`
   - Used for both development evaluation and final evaluation
   - Development usage example:
     ```python
     import requests

     resp = requests.post(
         "http://evaluation-api:8080/evaluate/agent",
         json={
             "agent_file": "/workspace/agent.py",
             "split": "eval",
             "first_k": 20,  # optional: evaluate only the first 20 problems for fast iteration
             "timeout": 3600,
             "case_timeout": 10,  # must be 10s
         },
         timeout=3900,  # client-side timeout (time the client waits), must be greater than json.timeout (at least +300s)
     )
     print(resp.json())
     ```
   - Note: `requests.post(..., timeout=...)` is the client-side timeout; `json.timeout` is the server-side timeout for running the agent
   - Note: The client-side timeout must always be greater than the server-side `json.timeout`, and the two should differ by at least `300` seconds
   - Note: `case_timeout` must be fixed at `10` (seconds)
   - Note: `first_k` is only supported with `split=eval`; it takes the first `k` problems in the original order of `lcb_eval.jsonl` — both the generation and scoring phases process only these first `k` problems
   - Concurrency limit: `/evaluate/agent` is **single-run** (only one evaluation can run at a time). If another evaluation is running, you will receive HTTP `409` with the error message `"another eval is running"`. To stop the currently running evaluation (and clean up stuck processes), call:
     ```python
     import requests
     requests.post('http://evaluation-api:8080/evaluate/agent', json={'kill_running': True})
     ```
     You can also include `kill_running: true` in the same JSON body as `agent_file` to achieve "kill then run" in a single request. If you still receive `409`, retry your evaluation request.
   - Security note: Inside the main container, `TASK_MODEL_API_BASE` points to the evaluation container's local proxy endpoint (`/v1/chat/completions`); the proxy enforces `TASK_MODEL_NAME` and isolates the real upstream `TASK_MODEL_API_KEY`
   - Security note: `split=test` data is stored inside the container as chunked encrypted files (`.enc`); they are not read into memory as whole files at build time — they are temporarily decrypted to `/tmp` only upon receiving a valid `X-Verifier-Secret`, and are automatically cleaned up after the request ends
   - Debugging note: When an agent fails or times out, the response will contain `agent_output` (stdout) and `agent_stderr` (stderr) fields; these should be the first place to look when troubleshooting
   - Return value note: A successful response returns evaluation fields such as `accuracy`, `correct`, `total`, `covered`, `coverage`, `scores`, and `detailed_results`; where `accuracy = correct / total * 100` and `coverage = covered / total * 100`; the API will not return `predictions_content` or any complete prediction code content
   - Typical timeout response example:
     ```json
     {
       "success": false,
       "error": "Agent execution timed out (limit: 3600s)",
       "agent_output": "...",
       "agent_stderr": "..."
     }
     ```
   - `split=test` can only be called by the verifier (you do not need to call it manually)

---

## Input / Output Contract

### Input Format

Each problem is a JSONL entry:

```json
{
  "idx": "abc344_d",
  "question_title": "...",
  "question_content": "...",
  "starter_code": "...",
  "platform": "leetcode|codeforces|atcoder",
  "difficulty": "easy|medium|hard",
  "contest_date": "2024-11-16T00:00:00",
  "question_id": "...",
  "contest_id": "...",
  "fn_name": "optional"
}
```

- `fn_name != null`: function-call style judging
- `fn_name == null`: stdin/stdout style judging

### Output Format

You need to output a JSONL prediction file (saved automatically by the runner), one entry per problem:

```json
{
  "idx": "abc344_d",
  "pred": "<python_code_string>"
}
```

- `idx` uses the upstream original `question_id` string
- `pred` must be a single code string (only one final answer may be submitted)

---

## Implementation Requirements

- You must implement an Agent class that inherits `BaseLCBAgent` in `/workspace/agent.py`
- Base class path: `/app/tools/base_agent.py`
- To avoid import identity conflicts, use the following fixed import style in `agent.py`:
  ```python
  import sys
  sys.path.insert(0, "/app/tools")
  from base_agent import BaseLCBAgent, Problem, Prediction
  from openai_helper import OpenAIHelper
  ```
- Do not use `from tools.base_agent ...` or `from app.tools...`
- Must implement:
  - `solve(problems, timeout_sec) -> List[Prediction]`
- Do not hardcode eval data (the final evaluation uses a different test set)

## Dependency Installation

If additional dependencies are needed, declare them in `/workspace/pyproject.toml`. The evaluation API will install them automatically before executing the agent.

---

## Development vs Submission

### Development (you can do this actively)

1. Implement and iterate in `/workspace/agent.py`
2. Call `/evaluate/agent` with `split=eval` to get feedback; add `first_k` for fast iteration
3. Perform error analysis and improvements based on `detailed_results`

### Final Submission (executed automatically by the system)

- The verifier will call the same API and the same agent file on `split=test`
- The verifier's total outer time limit is: `46800` seconds (13 hours)
- Default parameters for the final evaluation: `timeout=43200`, `case_timeout=10`
- You do not need to submit a test request manually
- Therefore your strategy must be generalizable — it must not rely on memorizing eval problems

---

## Evaluation Metrics

Common fields in the evaluation response (subject to the current implementation):
- `accuracy`: accuracy across the full split (`correct / total * 100`)
- `correct`: number of problems actually answered correctly
- `total`: total number of problems in the split
- `covered`: number of problems that were actually produced and evaluated within the timeout
- `coverage`: coverage rate (`covered / total * 100`)
- `scores`: per-problem boolean pass list, only for covered problems
- `detailed_results`: per-problem detailed results, only for covered problems
- The API will not return `predictions_content` or complete prediction code via HTTP

---

## Time Budget

You have a 12-hour (43,200-second) compute budget. Recommended approach:
- First build a working baseline
- Then iterate in small, fast steps
- Prioritize improving the quality of each single final answer (to raise `accuracy`)

---

## Recommended Workflow

1. Implement a `BaseLCBAgent` subclass in `/workspace/agent.py`
2. When constructing the prompt for the OpenAI model, explicitly include the mandatory output rules for `starter_code` / stdin
3. If you need to extend helpers, create a new filename and explicitly `import` it in `agent.py` (do not modify `/app/tools`)
4. Continuously iterate using `split=eval`
5. Perform error analysis by platform / problem type (`fn_name` vs stdin)
6. Continuously optimize overall `accuracy`

## Tips

1. First ensure the output format is stable (to avoid invalid evaluations)
2. For function-call problems, prioritize preserving the function signature and class structure
3. For stdin problems, prioritize ensuring a complete IO loop
4. Submit only one code answer per problem; you may use repeated sampling strategies — the final answer can be selected from candidates or synthesized from multiple candidates as the best code
5. Note that complex workflows require rate limiting, concurrency control, and retry backoff (e.g. exponential backoff)
6. After each change, validate the improvement using eval results
7. You can call `TASK_MODEL_API_BASE` directly in the main container

Good luck getting a higher score.
