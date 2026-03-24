# Mind2Web-2 Evaluator (Green Agent)

Evaluates web search agents on the [Mind2Web-2](https://github.com/OSU-NLP-Group/Mind2Web-2) benchmark using an LLM-as-a-judge pattern with hierarchical verification trees.

This is the **green agent** (evaluator) in the AgentBeats Mind2Web-2 scenario. It sends research tasks to a purple agent, receives markdown answers with URL citations, then scores them using the Mind2Web-2 evaluation pipeline (extraction, URL verification via Chromium, scoring).

## How It Works

1. Receives an eval request with a purple agent URL and config
2. Discovers eval scripts from the Mind2Web-2 dataset (filtered by domain CSV)
3. For each task (concurrently):
   - Extracts `TASK_DESCRIPTION` from the eval script
   - Sends it to the purple agent via A2A
   - Runs `evaluate_answer()` which builds a verification tree, extracts structured data from the answer, and verifies claims against cited URLs using Chromium
4. Aggregates scores and returns results

## Project Structure

```
src/
├─ server.py       # A2A server and agent card
├─ executor.py     # A2A request handling
├─ agent.py        # Evaluation orchestration
├─ llm_client.py   # LiteLLM adapter for mind2web2
└─ messenger.py    # A2A inter-agent communication
Dockerfile             # Docker build (includes Chromium + HF dataset)
pyproject.toml         # Dependencies (includes mind2web2 from git)
amber/
├─ amber-manifest-green.json5   # Green agent Amber manifest
├─ amber-manifest-purple.json5  # Purple agent Amber manifest
└─ amber-scenario.json5         # Full scenario for Amber
test_run.py            # Test script to run evaluation locally
```

## Configuration

The eval request JSON supports these config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `domain` | `"dev_set"` | Task set: `dev_set` (10 tasks) or `test_set` (314 tasks) |
| `num_tasks` | all | Limit number of tasks to run |
| `task_ids` | all | Run only specific task IDs |
| `judge_model` | `AGENT_LLM` env | LLM model for the judge (litellm format) |
| `max_concurrent` | `10` | Max tasks to run in parallel |

## Running Locally

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- A HuggingFace account with access to the [osunlp/Mind2Web-2](https://huggingface.co/datasets/osunlp/Mind2Web-2) gated dataset

### Setup

```bash
uv sync

# Download eval scripts from HuggingFace
HF_TOKEN=hf_... uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('osunlp/Mind2Web-2', repo_type='dataset', local_dir='mind2web2-data', token='YOUR_TOKEN')
"

# Install Chromium for URL verification
uv run patchright install chromium
```

### Run

Start the purple agent first (see [agent-template](https://github.com/RDI-Foundation/agent-template)), then:

```bash
AGENT_LLM=gemini/gemini-2.0-flash GEMINI_API_KEY=... uv run src/server.py --port 9009
```

### Test

```bash
uv run test_run.py
```

## Running with Docker

```bash
docker build --build-arg HF_TOKEN=hf_... -t mind2web2-green .
docker run -p 9009:9009 -e AGENT_LLM=gemini/gemini-2.0-flash -e GEMINI_API_KEY=... mind2web2-green
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENT_LLM` | yes | LLM model in litellm format (e.g. `gemini/gemini-2.0-flash`) |
| `GEMINI_API_KEY` | if using Gemini | Gemini API key |
| `OPENAI_API_KEY` | if using OpenAI | OpenAI API key |
| `ANTHROPIC_API_KEY` | if using Anthropic | Anthropic API key |
| `DEEPSEEK_API_KEY` | if using DeepSeek | DeepSeek API key |
| `DATA_DIR` | no (default: `mind2web2-data`) | Path to Mind2Web-2 dataset |
| `CACHE_DIR` | no (default: `cache`) | Path for evaluation cache |
