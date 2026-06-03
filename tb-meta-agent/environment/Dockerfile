FROM python:3.11-slim

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    procps \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Python deps:
# - flask/requests for the local proxy evaluation API
# - openai for agent-authored code (optional but consistent with other tasks)
RUN pip install --no-cache-dir \
    flask>=2.0.0 \
    requests>=2.31.0 \
    openai>=1.0.0 \
    anthropic

RUN mkdir -p /app/tools /evaluation_logs /logs/verifier

COPY tools/base_agent.py /app/tools/base_agent.py
COPY tools/anthropic_helper.py /app/tools/anthropic_helper.py
COPY tools/harbor/ /app/tools/harbor/

ENV PYTHONPATH=/app/tools:$PYTHONPATH
ENV WORKSPACE=/workspace

COPY environment/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
