# AIME Meta-Agent Optimization Task

## ⚠️ CRITICAL RESTRICTIONS ⚠️

**YOU MAY ONLY USE THE PROVIDED API ENDPOINT**

You are **strictly prohibited** from:
- ❌ Using any other LLM APIs (Anthropic, Google, Cohere, etc.)
- ❌ Running local models (Ollama, llama.cpp, vLLM, etc.)
- ❌ Accessing pre-trained models from HuggingFace, ModelScope, etc.
- ❌ Downloading/Accessing AIME test sets from the web.
- ❌ Using any AI services not explicitly provided.

**You may ONLY use:**
- ✅ The OpenAI-compatible API endpoint provided via environment variables
- ✅ Your own logic and algorithms

**Violation of these restrictions will result in immediate disqualification.**

Network monitoring is in place to detect unauthorized API usage.

---

## Objective

Your goal is to **maximize performance on the AIME (American Invitational Mathematics Examination) math benchmark** by building, testing, and optimizing agent workflows or prompts.

## Task Description

You need to create an agent/workflow system that can solve AIME math problems. You can:
- Design prompting strategies (CoT, few-shot, etc.)
- Build multi-step reasoning workflows
- Implement self-correction loops
- Use tool-augmented approaches
- Experiment with different decomposition strategies

## Environment Setup

### Available Tools

1. **OpenAI API Endpoint** (`/app/tools/openai_helper.py`)
   - Provides an OpenAI-compatible interface
   - Call models via Python API
   - Environment variable: `TASK_MODEL_API_BASE` and `TASK_MODEL_API_KEY`

2. **Model README** (`/workspace/model_readme.md`)
   - The README of the model you are using, from its HuggingFace repository
   - Contains model capabilities, recommended usage patterns, etc.
   - Read this file first to understand the model's strengths and optimize your prompting strategy

3. **Math Evaluation API** (Auto-started)
   - **Unified interface for development and submission**
   - Submit your agent file, get instant feedback
   - **Usage:**
     ```python
     import requests

     # Test your agent during development (eval split)
     response = requests.post('http://evaluation-api:8080/evaluate/agent', json={
         'agent_file': '/workspace/my_agent.py',
         'split': 'eval',
         'timeout': 21600
     })
     results = response.json()
     print(f"Accuracy: {results['accuracy']:.1f}%")

     # For final submission, the system will call with split='test'
     # (You don't need to do this manually)
     ```
   - **Returns:**
     - `success`: Boolean
     - `accuracy`: Percentage (0-100)
     - `correct`: Number correct
     - `total`: Total problems
     - `scores`: Per-problem correctness

> **Tip:** You can pass `'first_k': N` to `/evaluate/agent` to only evaluate the first N problems. This is useful for fast iteration. Default is `None` (evaluate all).

> **Concurrency note:** `/evaluate/agent` is **single-running** (only one evaluation can run at a time). If another evaluation is running, you will get HTTP `409` with error `"another eval is running"`. To stop the currently running evaluation (and clean up any stuck processes), call:
> ```python
> import requests
> requests.post('http://evaluation-api:8080/evaluate/agent', json={'kill_running': True})
> ```
> You can also include `kill_running: true` in the same JSON body as `agent_file` to do a best-effort "kill then run" in a single request. Then retry your evaluation request if you still get `409`.

### Input Format

Each problem is a JSON line with:
```json
{
  "idx": 0,
  "question": "Problem statement..."
}
```

## Implementation Requirements

**You must implement your solution as an Agent class** that inherits from `BaseAIMEAgent`.

- Base class location: `/app/tools/base_agent.py`
- Required method: `solve(problems, timeout_sec)` - takes a list of Problem objects and returns predictions
- Save your agent to `/workspace/agent.py`

**Example:**
```python
from base_agent import BaseAIMEAgent, Problem, Prediction
from openai_helper import OpenAIHelper

class AIMEAgent(BaseAIMEAgent):
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
name = "aime-agent"
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
       'timeout': 21600 # default 6 hours; adjust as needed
   })
   ```
3. Iterate based on feedback to improve accuracy

**Submission** (Automatic):
- The evaluation system will call your agent with **test split**:
  ```python
  # This happens automatically - you don't do this
  response = requests.post('http://evaluation-api:8080/evaluate/agent', json={
      'agent_file': '/workspace/agent.py',
      'split': 'test',
      'timeout': 43200  # 12 hours for final test
  })
  ```
- Your agent receives different problems (test set)
- **Do NOT hardcode eval data** - your agent must work with any input

The evaluation system will call your agent through this standardized interface.

## Optimization Strategies

Here are some ideas to explore:

### Prompting Techniques
- **Chain-of-Thought (CoT)**: Ask model to show step-by-step reasoning
- **Few-shot learning**: Provide example solutions
- **Self-consistency**: Generate multiple solutions and vote
- **Least-to-most prompting**: Break down problems into subproblems

### Workflow Approaches
- **Multi-step reasoning**: Break problem → Solve steps → Combine
- **Verification loops**: Generate solution → Verify → Refine
- **Ensemble methods**: Multiple approaches, select best
- **Error analysis**: Study failures and adapt strategy

### Advanced Techniques
- **Code generation**: Convert math to executable code
- **Symbolic manipulation**: Use computer algebra
- **Search/tree exploration**: Explore solution space

## Evaluation

Your final submission will be scored on the **protected test set**.

**Metrics:**
- **Accuracy**: Percentage of correct predictions
- **Average Score**: Mean of individual problem scores

## Time Management

You have **12 hours (43,200 seconds)** of compute time. Use it wisely.

## Tips

1. **Start simple**: Get a basic working system first
2. **Measure everything**: Track accuracy after each change
3. **Analyze errors**: Understand where and why your system fails
4. **Iterate quickly**: Make small improvements continuously
5. **Validate format**: Ensure your output matches requirements exactly

Good luck! Build the best AIME-solving agent you can.
