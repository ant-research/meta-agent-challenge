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
    anthropic \
    numpy \
    scipy \
    sympy \
    pandas \
    matplotlib \
    jupyter \
    ipython

# Create necessary directories
RUN mkdir -p /workspace/data /app/tools /logs/verifier

# Only copy agent-facing tools (base class + API helper + search helper)
# Search credentials are NOT baked here — search_helper calls the
# evaluation-api proxy at http://127.0.0.1:8080 instead.
COPY tools/base_agent.py /app/tools/base_agent.py
COPY tools/openai_helper.py /app/tools/openai_helper.py
COPY tools/search_helper.py /app/tools/search_helper.py

# Set environment variables
ENV PYTHONPATH=/app/tools:$PYTHONPATH
ENV WORKSPACE=/workspace

# Default command
CMD ["/bin/bash"]
