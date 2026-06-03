# Science Meta-Agent Task

## ⚠️ CRITICAL RESTRICTIONS ⚠️

**YOU MAY ONLY USE THE PROVIDED API ENDPOINTS**

You are **strictly prohibited** from:
- ❌ Using any other LLM APIs (Anthropic, Google Gemini, Cohere, etc.)
- ❌ Running local models (Ollama, llama.cpp, vLLM, etc.)
- ❌ Using unauthorized search APIs (Bing, DuckDuckGo, SerpAPI, etc.)
- ❌ Downloading answers or datasets from the web.
- ❌ Using any AI services not explicitly provided.
- ❌ **Calling the search API directly** via `urllib`, `requests`, `httpx`, or any other HTTP library. You **MUST** use the provided `SearchHelper` class.

**You may ONLY use:**
- ✅ The OpenAI-compatible API endpoint provided via environment variables
- ✅ Web Search API **through `SearchHelper` only** (`from search_helper import SearchHelper`)
- ✅ Your own logic and algorithms

**Violation of these restrictions will result in immediate disqualification.**

---

## Objective

Your goal is to **maximize accuracy on graduate-level science questions** by building, testing, and optimizing agent workflows or prompts.

## Task Description

You need to create an agent that can solve multiple-choice science questions covering physics, chemistry, biology, and other scientific domains. You can:
- Design prompting strategies (CoT, few-shot, etc.)
- Build multi-step reasoning workflows
- Use web search to look up relevant information
- Implement self-correction loops
- Experiment with different decomposition strategies

## Environment Setup

### Available Tools

1. **OpenAI API Endpoint** (`/app/tools/openai_helper.py`)
   - Provides an OpenAI-compatible interface
   - Environment variables: `TASK_MODEL_API_BASE`, `TASK_MODEL_API_KEY`, `TASK_MODEL_NAME`

2. **Web Search API** (`from search_helper import SearchHelper`)
   - Search the web for information
   - API credentials are pre-configured — just import and use, no setup needed
   - **⚠️ You MUST use this helper** — direct HTTP calls to the search endpoint are forbidden and will be flagged as violations
   - **Call limit: 2,500 calls per split** (dev and test are counted separately)
   - Usage:
     ```python
     import sys
     sys.path.append('/app/tools')
     from search_helper import SearchHelper

     search = SearchHelper()
     data = search.search("quantum mechanics")
     ```
   - Check remaining quota:
     ```python
     stats = search.get_usage_stats()
     print(f"Remaining: {stats['remaining']}/{stats['limit']}")
     # Or:
     print(f"Remaining: {search.get_remaining_quota()}")
     ```
   - You can override default search parameters via kwargs:
     ```python
     data = search.search("quantum mechanics", country="uk", language="en")
     ```
   - When quota is exhausted, `search()` raises `SearchQuotaExceeded` — plan your search budget wisely

3. **Model README** (`/workspace/model_readme.md`)
   - Contains model capabilities and recommended usage patterns
   - Read this to understand the model's strengths

4. **Science Evaluation API** (Auto-started)
   - Submit your agent file, get instant feedback
   - **Usage:**
     ```python
     import requests

     # Test your agent during development (eval split)
     response = requests.post('http://evaluation-api:8080/evaluate/agent', json={
         'agent_file': '/workspace/agent.py',
         'split': 'eval',
         'timeout': 21600
     })
     results = response.json()
     evaluation = results['evaluation']
     print(f"Accuracy: {evaluation['accuracy']*100:.1f}%")
     print(f"Correct: {evaluation['correct']}/{evaluation['total']}")
     ```
   - **Returns:**
     - `success`: Boolean
     - `evaluation.accuracy`: Float (0.0–1.0)
     - `evaluation.correct`: Number correct
     - `evaluation.total`: Total problems

> **Tip:** You can pass `'first_k': N` to `/evaluate/agent` to only evaluate the first N problems. Useful for fast iteration.

> **Concurrency note:** `/evaluate/agent` is **single-running** (only one evaluation can run at a time). If another evaluation is running, you will get HTTP `409` with error `"another eval is running"`. To stop the currently running evaluation (and clean up any stuck processes), call:
> ```python
> import requests
> requests.post('http://evaluation-api:8080/evaluate/agent', json={'kill_running': True})
> ```
> You can also include `kill_running: true` in the same JSON body as `agent_file` to do a best-effort "kill then run" in a single request. Then retry your evaluation request if you still get `409`.

### Input Format

Each problem passed to your `solve()` method has:
```python
problem.idx      # int: question index
problem.question # str: question text
problem.choices  # List[str]: answer choices, e.g. ["A) ...", "B) ...", "C) ...", "D) ..."]
```

The dev set (eval split) may have a varying number of answer choices (often 5 or more). The test set has exactly 4 choices (A, B, C, D).

## Implementation Requirements

**You must implement your solution as an Agent class** that inherits from `BaseScienceAgent`.

- Base class location: `/app/tools/base_agent.py`
- Required method: `solve(problems, timeout_sec)` — takes a list of Problem objects and returns predictions
- Save your agent to `/workspace/agent.py`

**Example:**
```python
from base_agent import BaseScienceAgent, Problem, Prediction
from openai_helper import OpenAIHelper

class ScienceAgent(BaseScienceAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your initialization here
        ...

    def solve(self, problems: list[Problem], timeout_sec: int) -> list[Prediction]:
        # Your implementation here
        ...
```

### Dependencies

If your agent needs additional Python packages, create a `/workspace/pyproject.toml`:

```toml
[project]
name = "science-agent"
version = "0.1.0"
dependencies = [
    "requests",
    "some-package>=1.0",
]
```

The evaluation system will automatically install your dependencies before running your agent.

### Development vs. Submission

**Development Phase** (12 hours):
1. Implement your Agent class in `/workspace/agent.py`
2. Test using the evaluation API with **eval split**:
   ```python
   import requests
   response = requests.post('http://evaluation-api:8080/evaluate/agent', json={
       'agent_file': '/workspace/agent.py',
       'split': 'eval',
       'timeout': 21600
   })
   ```
3. Iterate based on feedback to improve accuracy

**Submission** (Automatic):
- The verifier will call the same API (`/evaluate/agent`) with your `/workspace/agent.py` on `split=test` — different questions with exactly 4 choices (A, B, C, D)
- Default parameters for the final evaluation: `timeout=43200`
- If you have a `/workspace/pyproject.toml`, dependencies will be installed automatically before running your agent
- You do not need to submit a test request manually
- **Do NOT hardcode eval answers** — your agent must work with any input

## Optimization Strategies

### Prompting Techniques
- **Chain-of-Thought (CoT)**: Ask model to show step-by-step reasoning
- **Few-shot learning**: Provide example solutions
- **Self-consistency**: Generate multiple solutions and vote
- **Least-to-most prompting**: Break down problems into subproblems

### Search-Augmented Reasoning
- Use web search to look up specific facts, equations, or definitions
- Retrieve relevant context before answering
- Combine search results with LLM reasoning

### Workflow Approaches
- **Multi-step reasoning**: Identify topic → Search for info → Reason → Answer
- **Verification loops**: Generate answer → Verify → Refine
- **Ensemble methods**: Multiple approaches, select best

## Restrictions

### Allowed APIs
- ✅ LLM API at `TASK_MODEL_API_BASE` (provided)
- ✅ Web Search API **via `SearchHelper` only** (2,500 calls per split)

### Blocked APIs
- ❌ Other LLM APIs (Anthropic, Cohere, Hugging Face, etc.)
- ❌ Other search APIs (Bing, DuckDuckGo, SerpAPI, etc.)
- ❌ Local models (Ollama, llama.cpp, vLLM, etc.)
- ❌ Direct HTTP calls to the search endpoint (urllib, requests, httpx, etc.) — use SearchHelper

Violations will result in automatic failure.

## Scoring

Your final submission will be scored on the **protected test set** (GPQA dataset, 198 questions).

**Metrics:**
- **Accuracy**: Fraction of correct predictions (0.0–1.0)
- **Reward**: Equal to accuracy

## Time Management

You have **12 hours (43,200 seconds)** of compute time. Use it wisely.

## Tips

1. **Start simple**: Get a basic working system first, then iterate
2. **Measure everything**: Track accuracy after each change with `split="eval"`
3. **Use search**: Science questions often require specific factual knowledge
4. **Analyze errors**: Understand where and why your system fails
5. **Handle timeouts**: Implement timeout checking to return partial results

Good luck! Build the best science-solving agent you can.
