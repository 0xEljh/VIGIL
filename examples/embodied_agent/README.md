# Embodied Agent Demo

This example shows how to compose VIGIL with a detector loop, local LLM inference, tool calling, and an interactive Textual UI.

## Install

From the repository root:

```bash
uv sync --all-extras
```

If you only want demo/runtime dependencies:

```bash
uv sync --extra examples
```

## Run Pipeline

Batch mode:

```bash
uv run python -m examples.embodied_agent.orchestration.pipeline --quantization Q4_K_M
```

Interactive UI mode:

```bash
uv run python -m examples.embodied_agent.orchestration.pipeline --quantization Q4_K_M --interactive-user-input
```

## WebSocket Frame Ingest

Start the pipeline in API frame mode:

```bash
uv run python -m examples.embodied_agent.orchestration.pipeline --quantization Q4_K_M --frame-source-mode api
```

In another terminal, stream webcam frames:

```bash
uv run python examples/embodied_agent/send_webcam_ws.py --url ws://127.0.0.1:8000/ws/video
```

## API Server Only

```bash
bash examples/embodied_agent/setup_api_server.sh
```

This serves `examples.embodied_agent.api.main:app` with `/ws/video`, `/health`, and debug frame endpoints.
