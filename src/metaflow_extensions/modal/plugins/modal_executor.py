"""Modal executor — launches Metaflow steps in Modal sandboxes.

Layer: Execution
May only import from: metaflow stdlib, modal SDK

This module handles the Metaflow-specific concerns for Modal execution:
- mflog structured log capture (export_mflog_env_vars, bash_capture_logs, BASH_SAVE_LOGS)
- Metaflow environment variable assembly (DEFAULT_METADATA, SERVICE_INTERNAL_URL, config_values)
- Modal Sandbox lifecycle management (create, stream logs, wait, terminate)
- GPU type resolution from string specs to Modal GPU config objects

Mirrors ``metaflow.plugins.aws.batch.batch.Batch`` in structure.
"""

from __future__ import annotations

import os
import shlex
import sys
import threading
from typing import Any
from typing import Callable

from metaflow import util
from metaflow.exception import MetaflowException
from metaflow.metaflow_config import DEFAULT_METADATA
from metaflow.metaflow_config import SERVICE_INTERNAL_URL
from metaflow.mflog import BASH_SAVE_LOGS
from metaflow.mflog import bash_capture_logs
from metaflow.mflog import export_mflog_env_vars

# Redirect structured logs to $PWD/.logs/
LOGS_DIR = "$PWD/.logs"
STDOUT_FILE = "mflog_stdout"
STDERR_FILE = "mflog_stderr"
STDOUT_PATH = os.path.join(LOGS_DIR, STDOUT_FILE)
STDERR_PATH = os.path.join(LOGS_DIR, STDERR_FILE)

_INSTALL_HINT = (
    "Modal SDK not found. Install it with:\n"
    "\n"
    "    pip install metaflow-modal\n"
    "\n"
    "Then configure Modal credentials:\n"
    "    modal token new\n"
    "\n"
    "Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.\n"
    "See https://modal.com/docs for details."
)

# Cloud credential env vars to forward into the Modal sandbox.
_FORWARDED_CREDENTIAL_VARS = [
    # AWS
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_DEFAULT_REGION",
    # GCS
    "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_CONFIG",
    # Azure
    "AZURE_STORAGE_CONNECTION_STRING",
    "AZURE_STORAGE_KEY",
    # Modal auth (needed in sandbox for nested Modal calls)
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "MODAL_ENVIRONMENT",
]

_DEFAULT_PYTHON_VERSION = "3.11"


class ModalException(MetaflowException):
    headline = "Modal execution error"


class ModalExecutor:
    """Create a Modal sandbox and run a Metaflow step inside it.

    Follows the same structure as ``metaflow.plugins.aws.batch.batch.Batch``.

    The executor uses Modal.Sandbox.create() to spin up an ephemeral sandbox,
    runs the full Metaflow step command inside it (with mflog log capture),
    streams stdout/stderr back to the local terminal, and cleans up on exit.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment
        self._sandbox: Any = None
        self._on_log: Callable[[str, str], None] | None = None

    # ------------------------------------------------------------------
    # Image construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_image(image: str | None) -> Any:
        """Build a Modal Image from a Docker registry URL.

        If *image* is None, uses Modal's optimized debian_slim base image
        with Python 3.11.  Otherwise wraps the user-provided Docker Hub /
        registry URL in ``modal.Image.from_registry()``.
        """
        try:
            import modal
        except ImportError:
            raise ImportError(_INSTALL_HINT) from None

        if image is None:
            return modal.Image.debian_slim(python_version=_DEFAULT_PYTHON_VERSION)

        # User-specified Docker image URL (e.g. "python:3.11-slim",
        # "ghcr.io/myorg/myimage:latest")
        return modal.Image.from_registry(image)

    # ------------------------------------------------------------------
    # GPU resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gpu(gpu: str | None) -> Any:
        """Convert a GPU spec string to a Modal GPU configuration object.

        Supports format ``"TYPE"`` or ``"TYPE:COUNT"`` (e.g. ``"A10G:2"``).
        Unknown types are passed through as-is so that Modal can raise a
        descriptive error rather than us silently discarding the request.
        """
        if not gpu:
            return None

        try:
            import modal
        except ImportError:
            return gpu  # ImportError surfaced later in launch()

        parts = gpu.upper().replace("-", "").split(":", 1)
        gpu_type = parts[0]
        count = int(parts[1]) if len(parts) > 1 else 1

        _GPU_MAP: dict[str, Any] = {
            "T4": modal.gpu.T4,
            "A10G": modal.gpu.A10G,
            "A100": modal.gpu.A100,
            "A10040GB": modal.gpu.A100,
            "A10080GB": getattr(modal.gpu, "A100_80GB", modal.gpu.A100),
            "H100": modal.gpu.H100,
            "L4": modal.gpu.L4,
        }

        cls = _GPU_MAP.get(gpu_type)
        if cls is not None:
            return cls(count=count) if count > 1 else cls()

        # Unknown type — pass through; Modal will raise a clear error.
        return gpu

    # ------------------------------------------------------------------
    # Environment variable assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _build_env(
        code_package_metadata: str,
        code_package_sha: str,
        code_package_url: str,
        datastore_type: str,
        sandbox_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Assemble the full set of environment variables for the sandbox."""
        proxy_service_metadata = DEFAULT_METADATA == "service"
        env: dict[str, str] = {
            "METAFLOW_CODE_METADATA": code_package_metadata,
            "METAFLOW_CODE_SHA": code_package_sha,
            "METAFLOW_CODE_URL": code_package_url,
            "METAFLOW_CODE_DS": datastore_type,
            "METAFLOW_USER": util.get_username(),
            "METAFLOW_DEFAULT_DATASTORE": datastore_type,
            "METAFLOW_DEFAULT_METADATA": (
                "local" if proxy_service_metadata else DEFAULT_METADATA
            ),
            "METAFLOW_MODAL_WORKLOAD": "1",
        }

        if SERVICE_INTERNAL_URL:
            env["METAFLOW_SERVICE_URL"] = SERVICE_INTERNAL_URL

        from metaflow.metaflow_config_funcs import config_values

        for k, v in config_values():
            if (
                k.startswith("METAFLOW_DATASTORE_SYSROOT_")
                or k.startswith("METAFLOW_DATATOOLS_")
                or k.startswith("METAFLOW_S3")
                or k.startswith("METAFLOW_CARD_S3")
                or k.startswith("METAFLOW_CONDA")
                or k.startswith("METAFLOW_SERVICE")
            ):
                env[k] = v

        for var in _FORWARDED_CREDENTIAL_VARS:
            val = os.environ.get(var)
            if val:
                env[var] = val

        if sandbox_env:
            env.update(sandbox_env)

        return env

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _command(
        self,
        code_package_metadata: str,
        code_package_url: str,
        step_name: str,
        step_cmds: list[str],
        task_spec: dict[str, str],
        datastore_type: str,
    ) -> str:
        """Build the full bash command that runs inside the Modal sandbox.

        Structure:
        1. Set mflog environment variables
        2. Create log directory
        3. Download + extract code package (via get_package_commands)
        4. Bootstrap environment (via bootstrap_commands: conda/pypi)
        5. Execute the step with log capture
        6. Save logs and propagate exit code
        """
        mflog_expr = export_mflog_env_vars(
            datastore_type=datastore_type,
            stdout_path=STDOUT_PATH,
            stderr_path=STDERR_PATH,
            **task_spec,
        )

        init_cmds = self._environment.get_package_commands(
            code_package_url, datastore_type, code_package_metadata
        )
        # Avoid a directory named "metaflow" that would shadow the installed package.
        init_cmds = [
            cmd.replace(
                "mkdir metaflow && cd metaflow", "mkdir mf_modal && cd mf_modal"
            )
            for cmd in init_cmds
        ]
        init_expr = " && ".join(init_cmds)

        bootstrap_cmds = self._environment.bootstrap_commands(step_name, datastore_type)
        step_expr = bash_capture_logs(" && ".join(bootstrap_cmds + step_cmds))

        cmd_str = (
            f"true && mkdir -p {LOGS_DIR} && {mflog_expr} && {init_expr} && {step_expr}; "
            f"c=$?; {BASH_SAVE_LOGS}; exit $c"
        )

        # Metaflow's get_package_commands() uses \\" escaping designed for
        # the shlex round-trip in batch.py: shlex.split('bash -c "%s"' % cmd).
        # Apply the same transformation so escaped quotes resolve correctly.
        cmd_str = shlex.split(f'bash -c "{cmd_str}"')[-1]

        return cmd_str

    # ------------------------------------------------------------------
    # Launch + wait
    # ------------------------------------------------------------------

    def launch(
        self,
        step_name: str,
        step_cli: str,
        task_spec: dict[str, str],
        code_package_metadata: str,
        code_package_sha: str,
        code_package_url: str,
        datastore_type: str,
        image: str | None = None,
        cpu: float = 1.0,
        memory: int = 1024,
        gpu: str | None = None,
        timeout: int = 600,
        env: dict[str, str] | None = None,
        secrets: list[str] | None = None,
        on_log: Callable[[str, str], None] | None = None,
    ) -> None:
        """Create a Modal sandbox and run the step command inside it."""
        try:
            import modal
        except ImportError:
            raise ImportError(_INSTALL_HINT) from None

        cmd_str = self._command(
            code_package_metadata,
            code_package_url,
            step_name,
            [step_cli],
            task_spec,
            datastore_type,
        )

        sandbox_env = self._build_env(
            code_package_metadata,
            code_package_sha,
            code_package_url,
            datastore_type,
            sandbox_env=env,
        )

        modal_image = self._build_image(image)
        modal_gpu = self._resolve_gpu(gpu)

        modal_secrets = []
        for secret_name in (secrets or []):
            modal_secrets.append(modal.Secret.from_name(secret_name))

        # Inject METAFLOW_MODAL_SANDBOX_ID from Modal's task identifier (if available).
        run_cmd = (
            "export METAFLOW_MODAL_SANDBOX_ID=${MODAL_TASK_ID:-unknown} && "
            + cmd_str
        )

        self._sandbox = modal.Sandbox.create(
            "bash",
            "-c",
            run_cmd,
            image=modal_image,
            timeout=timeout,
            env=sandbox_env,
            cpu=float(cpu),
            memory=memory,
            gpu=modal_gpu,
            secrets=modal_secrets,
        )
        self._on_log = on_log

    def cleanup(self) -> None:
        """Terminate the sandbox if it is still running. Best-effort, never raises."""
        import contextlib

        if self._sandbox is not None:
            with contextlib.suppress(Exception):
                self._sandbox.terminate()
            self._sandbox = None

    def wait(self, echo: Any) -> None:
        """Stream sandbox output, wait for completion, and propagate exit code.

        Stdout and stderr are always drained via background threads regardless
        of whether on_log is set.  This avoids two hazards:

        (a) Reading a stream after sandbox.wait() may silently return empty
            because the underlying network connection is already closed by the
            time wait() returns.
        (b) Reading stdout then stderr sequentially would block on stdout while
            stderr produces output, losing ordering and potentially deadlocking.

        When on_log is None, lines are collected in memory and printed via
        echo() after wait() returns.

        The method calls sys.exit(exit_code) on non-zero exit — same contract
        as SandboxExecutor.wait().
        """
        if self._sandbox is None:
            raise ModalException("No sandbox — was launch() called?")

        sandbox = self._sandbox
        on_log = self._on_log

        if on_log is not None:
            def _read_stdout() -> None:
                try:
                    for line in sandbox.stdout:
                        on_log(line.rstrip("\n"), "stdout")
                except Exception:
                    pass

            def _read_stderr() -> None:
                try:
                    for line in sandbox.stderr:
                        on_log(line.rstrip("\n"), "stderr")
                except Exception:
                    pass

        else:
            # Collect into lists; print after wait().
            stdout_lines: list[str] = []
            stderr_lines: list[str] = []

            def _read_stdout() -> None:
                try:
                    for line in sandbox.stdout:
                        stdout_lines.append(line.rstrip("\n"))
                except Exception:
                    pass

            def _read_stderr() -> None:
                try:
                    for line in sandbox.stderr:
                        stderr_lines.append(line.rstrip("\n"))
                except Exception:
                    pass

        t_out = threading.Thread(target=_read_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_out.start()
        t_err.start()
        # raise_on_termination=False: handle preemption (returncode=137) and timeout
        # (returncode=124) via the normal exit-code path below instead of raising.
        try:
            sandbox.wait(raise_on_termination=False)
        except Exception:
            # SandboxTimeoutError still fires even with raise_on_termination=False
            # (only affects TERMINATED, not TIMEOUT). _result is set before the raise,
            # so returncode=124 is available. Any other unexpected wait() error is also
            # caught here; returncode will be None and we default to 1 below.
            pass
        t_out.join(timeout=30)
        t_err.join(timeout=30)

        if t_out.is_alive() or t_err.is_alive():
            echo(
                "[modal] Warning: output stream did not close within 30s after "
                "sandbox exit; some output may be missing.",
                stream="stderr",
            )

        if on_log is None:
            for line in stdout_lines:
                echo(line, stream="stderr")
            for line in stderr_lines:
                echo(line, stream="stderr")

        exit_code = sandbox.returncode
        if exit_code is None:
            exit_code = 1  # wait() failed before result arrived (e.g. network error)
        self.cleanup()

        if exit_code != 0:
            echo(
                f"Modal task finished with exit code {exit_code}.",
                stream="stderr",
            )
            sys.exit(exit_code)
