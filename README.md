# metaflow-modal

[![CI](https://github.com/npow/metaflow-modal/actions/workflows/ci.yml/badge.svg)](https://github.com/npow/metaflow-modal/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/metaflow-modal)](https://pypi.org/project/metaflow-modal/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Status: WIP](https://img.shields.io/badge/status-WIP-orange.svg)](#) [![Docs](https://img.shields.io/badge/docs-mintlify-18a34a?style=flat-square)](https://mintlify.com/npow/metaflow-modal)

> **Work in progress.** The API is functional but not yet stable. Breaking changes may occur before 1.0.

Run your GPU-hungry Metaflow steps on Modal without managing any infrastructure.

## The problem

Running ML training or GPU inference inside a Metaflow pipeline means either paying for always-on GPU instances (expensive when idle) or wiring up a custom remote execution backend (time-consuming, fragile). The built-in `@batch` and `@kubernetes` decorators solve this for AWS and Kubernetes, but not for Modal's serverless GPU cloud — leaving a gap for teams who want on-demand GPU execution without the overhead.

## Quick start

```bash
pip install metaflow-modal
```

```python
from metaflow import FlowSpec, step, modal

class TrainFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.train)

    @modal(gpu="A10G", memory=8192)
    @step
    def train(self):
        import torch
        self.device = str(torch.cuda.get_device_name(0))
        self.next(self.end)

    @step
    def end(self):
        print(f"Trained on: {self.device}")

if __name__ == "__main__":
    TrainFlow()
```

```bash
METAFLOW_DEFAULT_DATASTORE=s3 python train_flow.py run
```

## Install

```bash
pip install metaflow-modal
```

Requires Modal credentials — run `modal token new` once to authenticate, or set:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

A remote datastore (S3, GCS, or Azure Blob) is required — `@modal` sandboxes can't reach the local filesystem.

## Usage

### CPU-only step with custom image

```python
@modal(cpu=4, memory=16384, image="python:3.11-slim", timeout=1800)
@step
def preprocess(self):
    ...
```

### GPU step with Modal secrets

```python
@modal(gpu="H100", secrets=["my-hf-token"])
@step
def finetune(self):
    import os
    token = os.environ["HF_TOKEN"]  # injected from Modal secret
    ...
```

### Multi-GPU

```python
@modal(gpu="A100:4")   # 4x A100 40GB
@step
def distributed_train(self):
    ...
```

### Supported GPUs

| Value | Hardware |
|-------|----------|
| `"T4"` | NVIDIA T4 |
| `"A10G"` | NVIDIA A10G |
| `"A100"` | NVIDIA A100 40GB |
| `"A100-80GB"` | NVIDIA A100 80GB |
| `"H100"` | NVIDIA H100 |
| `"L4"` | NVIDIA L4 |
| `"TYPE:N"` | N of TYPE (e.g. `"A10G:2"`) |

## How it works

`@modal` intercepts the Metaflow step execution and redirects it through Modal's sandbox API. Your flow's code package is uploaded to your remote datastore (S3/GCS/Azure), then downloaded and executed inside an ephemeral Modal container. Artifacts and metadata are written back to the same datastore, so the rest of your flow continues normally on your local machine.

```
@modal step  ->  upload code package to S3
             ->  modal.Sandbox.create("bash", "-c", step_cmd)
             ->  stream stdout/stderr live
             ->  sandbox exits  ->  sync artifacts + metadata from S3
```

No Modal Functions to deploy, no apps to register. Each step spins up a fresh container and exits.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cpu` | `1` | Number of CPUs |
| `memory` | `1024` | Memory in MB |
| `gpu` | `None` | GPU type (see table above) |
| `image` | Modal debian_slim | Docker image URL |
| `timeout` | `600` | Sandbox timeout in seconds |
| `env` | `{}` | Extra environment variables |
| `secrets` | `[]` | Modal secret names to mount |

| Environment variable | Purpose |
|---------------------|---------|
| `MODAL_TOKEN_ID` | Modal API token ID |
| `MODAL_TOKEN_SECRET` | Modal API token secret |
| `MODAL_ENVIRONMENT` | Modal environment name (optional) |
| `METAFLOW_DEFAULT_DATASTORE` | Must be `s3`, `gs`, or `azure` |

## Development

```bash
git clone https://github.com/npow/metaflow-modal
cd metaflow-modal
pip install -e ".[dev]"

# Unit tests (no credentials needed)
pytest tests/unit/

# Architecture enforcement tests
pytest tests/structural/ -m structural

# Lint + type check
ruff check src/ tests/
mypy src/
```

## License

[Apache 2.0](LICENSE)
