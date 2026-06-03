FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    procps \
    bc \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    openai>=1.0.0 \
    numpy \
    scipy \
    sympy \
    pandas \
    matplotlib \
    jupyter \
    ipython

# Create necessary directories
RUN mkdir -p /workspace/data /app/tools /logs/verifier

# Only copy agent-facing tools (base class + API helper)
COPY tools/base_agent.py /app/tools/base_agent.py
COPY tools/openai_helper.py /app/tools/openai_helper.py

# NOTE: solution/ is NOT copied here. Harbor's OracleAgent uploads it
# at runtime to /solution/. This prevents the agent from accessing
# golden_answers.jsonl during normal (non-oracle) runs.

# Set environment variables
ENV PYTHONPATH=/app/tools:$PYTHONPATH
ENV WORKSPACE=/workspace
# API credentials are injected at runtime via docker-compose environment,
# pointing to the evaluation-api model proxy (not the real upstream API).

# Default command
CMD ["/bin/bash"]
