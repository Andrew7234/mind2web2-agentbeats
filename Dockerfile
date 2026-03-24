FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libnss3 libatk-bridge2.0-0 libdrm2 libxcomposite1 libxdamage1 \
    libxrandr2 libgbm1 libasound2 libpango-1.0-0 libcairo2 libatspi2.0-0 \
    libxshmfence1 libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Install Chromium for patchright (webpage capture during evaluation)
RUN uv run patchright install chromium

# Download eval scripts from HuggingFace (requires HF_TOKEN at build time)
ARG HF_TOKEN
RUN uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('osunlp/Mind2Web-2', repo_type='dataset', local_dir='mind2web2-data', token='$HF_TOKEN')"

ENV DATA_DIR=/home/agent/mind2web2-data
ENV CACHE_DIR=/home/agent/cache

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "8081"]
EXPOSE 8081
