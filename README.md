# The Meta-Agent Challenge

[![Website](https://img.shields.io/badge/Website-meta--agent--challenge.com-blue)](https://meta-agent-challenge.com)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](https://arxiv.org/abs/2606.04455)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
[![GitHub](https://img.shields.io/github/stars/ant-research/meta-agent-challenge?style=social)](https://github.com/ant-research/meta-agent-challenge)

The first benchmark that asks code agents to build agents — then measures how good those agents are. See the [leaderboard](https://meta-agent-challenge.com) on our website and the [paper](https://arxiv.org/abs/2606.04455) for more details.

## What is the Meta-Agent Challenge?

The Meta-Agent Challenge (MAC) flips the usual benchmark setup. Instead of asking an AI to solve a task, MAC asks it to *build the agent that solves the task* — autonomously, end-to-end, inside a sealed dual-container sandbox under a wall-clock and API budget.

A meta-agent (e.g. Claude Code, Codex, Gemini-cli) reads the task, edits a Python `agent.py`, runs it against a development set, iterates on the feedback, and submits a final artifact. A held-out test set — visible only to the verifier injected after the budget expires — produces the score.

**Five domains, two phases each:**

- **Meta-AIME** — AIME 2022–2023 → AIME 2024–2025 (`aime-meta-agent/`)
- **Meta-GPQA** — HLE multiple-choice → GPQA Diamond (`science-meta-agent/`)
- **Meta-LiveCodeBench** — LiveCodeBench, disjoint split (`lcb-meta-agent/`)
- **Meta-SWE-Bench** — SWE-Bench Verified, disjoint split (`swe-meta-agent/`)
- **Meta-Terminal-Bench** — Terminal-Bench Pro → Terminal-Bench 2.0 (`tb-meta-agent/`)

## Quick Start

```bash
pip install harbor==0.3.0

# Fill in your credentials — each run script sources this file and aborts
# if a required variable is missing.
cp .env.example .env
# Edit .env with your preferred editor

bash scripts/aime_meta_agent_claude_code.sh
```

Each domain has three run scripts, one per scaffold:

| Domain | Claude Code | Codex | Gemini CLI |
|---|---|---|---|
| Meta-AIME | `aime_meta_agent_claude_code.sh` | `aime_meta_agent_codex.sh` | `aime_meta_agent_gemini.sh` |
| Meta-GPQA | `science_meta_agent_claude_code.sh` | `science_meta_agent_codex.sh` | `science_meta_agent_gemini.sh` |
| Meta-LiveCodeBench | `lcb_meta_agent_claude_code.sh` | `lcb_meta_agent_codex.sh` | `lcb_meta_agent_gemini.sh` |
| Meta-SWE-Bench | `swe_meta_agent_claude_code.sh` | `swe_meta_agent_codex.sh` | `swe_meta_agent_gemini.sh` |
| Meta-Terminal-Bench | `tb_meta_agent_claude_code.sh` | `tb_meta_agent_codex.sh` | `tb_meta_agent_gemini.sh` |

## Utility scripts

- `scripts/audit_all.sh` — audit every experiment root under `$PARENT` (default: repo root) and write `outputs/audit_<name>.json`.

## Contribute

We welcome contributions of new domains, scaffolds, and reproducible runs.

- **Add a domain** — drop a new `*-meta-agent/` directory following the layout of the existing five (dev/test data, `instruction.md`, `task.toml`, evaluation oracle, Dockerfiles for the dual-container setup).
- **Add a scaffold** — implement a Harbor agent plugin and add a run script under `scripts/` for at least one domain.
- **Add a model** — open an issue with your full job directory so we can validate the run and add the model to the leaderboard.

Open an issue first for anything non-trivial. PRs should keep the existing directory structure and pass the verifier (`tests/test.sh`) on the affected domain.

## Citation

```bibtex
@misc{lu2026metaagentchallengecurrentagents,
      title={The Meta-Agent Challenge: Are Current Agents Capable of Autonomous Agent Development?}, 
      author={Xinyu Lu and Tianshu Wang and Pengbo Wang and zujie wen and Zhiqiang Zhang and Jun Zhou and Boxi Cao and Yaojie Lu and Hongyu Lin and Xianpei Han and Le Sun},
      year={2026},
      eprint={2606.04455},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.04455}, 
}
```

## License

[Apache 2.0](LICENSE)
