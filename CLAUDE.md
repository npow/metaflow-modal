# metaflow-modal

Run Metaflow steps on Modal's serverless GPU cloud.

## Architecture

Layers (dependencies flow downward only):

```
Decorator    — src/.../plugins/modal_decorator.py
    ↓
CLI + Exec   — src/.../plugins/modal_cli.py, modal_executor.py
    ↓
Modal SDK    — modal.Sandbox.create(...)
```

**Rule: no upward imports.** The executor must never import from the decorator or CLI.
The decorator must never import from the executor directly (it only redirects via CLI args).
Structural tests enforce this.

## Key files

| What | Where |
|------|-------|
| Metaflow decorator | `src/metaflow_extensions/modal/plugins/modal_decorator.py` |
| CLI command handler | `src/metaflow_extensions/modal/plugins/modal_cli.py` |
| Modal executor | `src/metaflow_extensions/modal/plugins/modal_executor.py` |
| Plugin registration | `src/metaflow_extensions/modal/plugins/__init__.py` |

## Execution flow

```
1. step_init()        — validate remote datastore, resolve resources
2. runtime_init()     — store flow, graph, package, run_id
3. runtime_task_created() — upload code package to datastore (once)
4. runtime_step_cli() — redirect: cli_args.commands = ["modal", "step"]

5. Metaflow runs subprocess:
   python flow.py modal step train <metadata> <sha> <url> --cpu=2 ...

6. modal_cli.py step():
   → build inner step command
   → create ModalExecutor
   → executor.launch():
     → build Modal Image (from_registry or debian_slim)
     → build env vars (mflog, datastore config, AWS creds, etc.)
     → modal.Sandbox.create("bash", "-c", cmd, ...)
   → executor.wait():
     → stream stdout/stderr via threads
     → sandbox.wait()
     → sandbox.returncode → sys.exit if non-zero

7. Inside Modal sandbox:
   → download + extract code package from S3
   → bootstrap environment (conda/pypi via bootstrap_commands)
   → python flow.py step train ...
   → artifacts saved to remote datastore
   → logs saved via mflog

8. task_pre_step() [inside sandbox] — emit modal-sandbox-id metadata
9. task_finished() [inside sandbox] — sync local metadata to datastore
```

## Commands

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/

# Unit tests (no credentials needed)
pytest tests/unit/

# Structural tests (enforce architecture)
pytest tests/structural/ -m structural

# Integration tests (need Modal + cloud datastore credentials)
MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... pytest tests/ -m integration
```

## Conventions

- **Lazy imports**: Never import `modal` at module top level. Import inside methods
  so that `import metaflow_extensions.modal` never triggers an ImportError when
  the modal SDK is not installed.
- **Error messages**: Every ImportError or configuration error must include an
  install command, required env vars, and a doc link.
- **Layer docstrings**: Every plugin file must have a module docstring with `Layer:`.
- **Remote datastore**: @modal requires s3/gcs/azure — same constraint as @batch.
  Local datastore raises ModalException in step_init().

## GPU types

Supported values for `@modal(gpu=...)`:

| Parameter | Modal GPU |
|-----------|-----------|
| `"T4"` | NVIDIA T4 |
| `"A10G"` | NVIDIA A10G |
| `"A100"` | NVIDIA A100 40GB |
| `"A100-80GB"` | NVIDIA A100 80GB |
| `"H100"` | NVIDIA H100 |
| `"L4"` | NVIDIA L4 |
| `"TYPE:N"` | N× of TYPE (e.g. `"A10G:2"`) |

## Environment variables

| Variable | Purpose |
|----------|---------|
| `MODAL_TOKEN_ID` | Modal API token ID |
| `MODAL_TOKEN_SECRET` | Modal API token secret |
| `MODAL_ENVIRONMENT` | Modal environment name (optional) |
| `METAFLOW_MODAL_TARGET_PLATFORM` | Override conda target arch (default: `linux-64`) |
| `METAFLOW_MODAL_WORKLOAD` | Set to `1` inside Modal sandbox |
| `METAFLOW_MODAL_SANDBOX_ID` | Sandbox ID (set from `$MODAL_TASK_ID` inside sandbox) |
