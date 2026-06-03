FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    procps \
    bc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    openai>=1.0.0 \
    requests \
    numpy \
    pandas \
    tqdm

RUN mkdir -p /app/tools /logs/verifier

COPY tools/base_agent.py /app/tools/base_agent.py
COPY tools/openai_helper.py /app/tools/openai_helper.py

ENV PYTHONPATH=/app/tools:$PYTHONPATH
ENV WORKSPACE=/workspace
# API configuration is injected at runtime via docker-compose.

CMD ["/bin/bash"]
