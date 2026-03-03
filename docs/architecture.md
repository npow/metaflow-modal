# Architecture

## Layer diagram

```
┌─────────────────────────────────┐
│  @modal decorator               │  ← Metaflow integration
│  modal_decorator.py             │
└──────────────┬──────────────────┘
               │ redirects via CLI args
┌──────────────▼──────────────────┐
│  CLI + Executor                 │  ← Remote execution
│  modal_cli.py, modal_executor.py│
└──────────────┬──────────────────┘
               │ uses
┌──────────────▼──────────────────┐
│  Modal SDK                      │  ← Provider API
│  modal.Sandbox.create(...)      │
└─────────────────────────────────┘
```

## Invariants

1. **Dependencies flow down only.** `modal_executor.py` must never import from
   `modal_decorator.py` or `modal_cli.py`.

2. **SDK imports are lazy.** `modal` is imported inside methods, not at module
   top level. This means `import metaflow_extensions.modal` never triggers an
   ImportError when `modal` is not installed.

3. **Error messages teach.** Every ImportError or configuration error includes
   the exact command to fix it (`pip install metaflow-modal`, `modal token new`,
   required env vars, doc link).

4. **Remote datastore required.** The Modal sandbox runs on Modal's cloud and
   cannot access the local filesystem. A remote datastore (S3, GCS, Azure Blob)
   is required — same as `@batch` and `@kubernetes`.

5. **Target platform is linux-64.** Modal always runs on x86_64 Linux. The
   decorator sets `target_platform = "linux-64"` so nflx-extensions resolves
   conda environments for the correct arch.

## Execution flow

### Local side (Metaflow runtime)

```
1. step_init()
   ├── validate remote datastore
   ├── cache Modal auth env vars into decorator attributes
   └── _ensure_conda_remote_command_aliases() (nflx-extensions compat)

2. runtime_init()
   └── store flow, graph, package, run_id

3. runtime_task_created()
   └── _save_package_once(): upload code tarball to S3/GCS/Azure (once per run)

4. runtime_step_cli()
   ├── cli_args.commands = ["modal", "step"]
   ├── append: package_metadata, package_sha, package_url
   ├── append: --cpu, --memory, --gpu, --image, --timeout
   ├── append: --env-var KEY=VALUE (repeated, includes Modal tokens)
   └── append: --secret NAME (repeated)

5. Metaflow forks subprocess:
   python flow.py modal step <step_name> <meta> <sha> <url> --cpu=2 --gpu=A10G ...
```

### CLI side (modal_cli.py → modal_executor.py)

```
6. step() click handler:
   ├── build entrypoint + top_args (flow-level CLI options)
   ├── build step_cli = "python flow.py ... step <step_name> ..."
   ├── collect env from @environment decorator
   ├── merge user --env-var KEY=VALUE overrides
   ├── forward MODAL_TOKEN_* into os.environ (needed by modal SDK)
   └── create ModalExecutor(environment)

7. executor.launch():
   ├── build bash command:
   │   ├── mkdir -p .logs && export MFLOG vars
   │   ├── get_package_commands() → download + extract code tarball from S3
   │   ├── bootstrap_commands() → install conda/pypi packages
   │   └── bash_capture_logs(step_cli) + BASH_SAVE_LOGS
   ├── build sandbox_env:
   │   ├── METAFLOW_CODE_*, METAFLOW_DEFAULT_*, METAFLOW_SERVICE_*
   │   ├── all METAFLOW_DATASTORE_SYSROOT_* / METAFLOW_S3* / METAFLOW_CONDA*
   │   └── cloud credentials (AWS_*, GOOGLE_*, AZURE_*, MODAL_*)
   ├── _build_image(image) → modal.Image.debian_slim() or from_registry()
   ├── _resolve_gpu(gpu) → modal.gpu.A10G() etc.
   └── modal.Sandbox.create("bash", "-c", run_cmd, image=..., env=..., ...)

8. executor.wait():
   ├── thread 1: for line in sandbox.stdout → on_log(line, "stdout")
   ├── thread 2: for line in sandbox.stderr → on_log(line, "stderr")
   ├── sandbox.wait()
   ├── exit_code = sandbox.returncode
   ├── cleanup() → sandbox.terminate() if still alive
   └── sys.exit(exit_code) if exit_code != 0
```

### Inside the Modal sandbox

```
9. bash -c "export METAFLOW_MODAL_SANDBOX_ID=... && ..."
   ├── mflog env vars set
   ├── download + extract code package (boto3/requests from S3)
   ├── bootstrap: install conda or pip packages
   └── python flow.py step <step_name> ...
       ├── task_pre_step(): emit modal-sandbox-id metadata
       ├── user step code runs
       ├── artifacts saved to S3/GCS/Azure
       └── task_finished(): sync local metadata to datastore
```

## Design decisions

### Why Modal Sandbox instead of Modal Functions?

Modal Functions require the function to be defined at module import time and
deployed in advance (or run via `modal run`). Modal Sandbox (`modal.Sandbox.create()`)
is ephemeral — it spins up a container, runs an arbitrary command, and exits. This
maps directly to Metaflow's `@batch` pattern of "run this bash command in a container".

### Why not depend on sandrun?

`sandrun` is the shared library underlying `metaflow-sandbox`. Since `metaflow-modal`
is a dedicated single-backend extension, a direct Modal SDK dependency is simpler:
no extra abstraction layer, no sandrun version coupling, and full access to
Modal-specific features (GPU types, secrets, volumes).

### Why debian_slim as the default image?

Modal's `debian_slim` is pre-cached on Modal's infrastructure, making cold starts
significantly faster than pulling `python:3.11-slim` from Docker Hub on every run.

### Why forward cloud credentials?

Modal sandboxes run on Modal's cloud infrastructure with no native IAM integration
for S3/GCS/Azure. We forward credentials from the user's local environment so the
sandbox can access the remote Metaflow datastore.

### Why two threads for log streaming?

`sandbox.stdout` and `sandbox.stderr` are blocking iterators. Reading them
sequentially would serialize stdout and stderr — logs would appear only after
the sandbox finishes. Two daemon threads let both streams flow concurrently.
