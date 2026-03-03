"""Metaflow step decorator for Modal execution.

Layer: Metaflow Integration
May only import from: metaflow stdlib

Provides the ``@modal`` decorator that executes Metaflow steps on Modal's
serverless GPU cloud. Follows the same lifecycle-hook pattern as ``@batch``
and ``@kubernetes``.

Usage:
    @modal(cpu=4, memory=16384, gpu="A10G")
    @step
    def my_step(self):
        ...
"""

from __future__ import annotations

import os
import sys
from importlib import import_module
from typing import Any
from typing import ClassVar

from metaflow.decorators import StepDecorator
from metaflow.exception import MetaflowException

_MODAL_AUTH_ENV_VARS = ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT")
_INITIAL_MODAL_AUTH_ENV = {
    k: v for k, v in ((var, os.environ.get(var)) for var in _MODAL_AUTH_ENV_VARS) if v
}
_MODAL_REMOTE_COMMAND_ALIASES = ("modal",)


def _ensure_conda_remote_command_aliases() -> None:
    """Ensure nflx-extensions conda remote-command checks include modal.

    ``CONDA_REMOTE_COMMANDS`` in nflx-extensions gates the target-arch
    selection in ``extract_merged_reqs_for_step``.  Without this patch,
    @modal steps would receive a native-arch (macOS) ``ResolvedEnvironment``
    instead of a ``linux-64`` one.
    """
    module_names = (
        "metaflow_extensions.netflix_ext.plugins.conda.conda_environment",
        "metaflow_extensions.netflix_ext.plugins.conda.conda_step_decorator",
    )
    for module_name in module_names:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        current = tuple(getattr(module, "CONDA_REMOTE_COMMANDS", ()))
        merged = tuple(dict.fromkeys([*current, *_MODAL_REMOTE_COMMAND_ALIASES]))
        module.CONDA_REMOTE_COMMANDS = merged


class ModalException(MetaflowException):
    headline = "Modal error"


class ModalDecorator(StepDecorator):
    """Run a Metaflow step on Modal's serverless GPU cloud.

    Resources are provisioned on demand — no idle infrastructure.
    Supports CPU, memory, and GPU (T4, A10G, A100, H100) configurations.

    Parameters
    ----------
    cpu : float
        Number of CPUs (default: 1).
    memory : int
        Memory in MB (default: 1024).
    gpu : str or None
        GPU type: "T4", "A10G", "A100", "A100-80GB", "H100", "L4".
        Default: None (no GPU).
    image : str or None
        Docker image URL (e.g. "python:3.11-slim"). Default: Modal's
        debian_slim image with Python 3.11.
    timeout : int
        Sandbox timeout in seconds (default: 600).
    env : dict
        Additional environment variables to set inside the sandbox.
    secrets : list[str]
        Names of Modal secrets to mount (e.g. ["my-aws-secret"]).
    """

    name = "modal"

    defaults: ClassVar[dict[str, Any]] = {
        "cpu": 1,
        "memory": 1024,
        "gpu": None,
        "image": None,
        "timeout": 600,
        "env": {},
        "secrets": [],
    }

    supports_conda_environment = True
    target_platform = os.environ.get("METAFLOW_MODAL_TARGET_PLATFORM", "linux-64")

    # Class-level code-package state (shared across all instances,
    # uploaded once per flow run — same pattern as BatchDecorator).
    package_metadata: ClassVar[str | None] = None
    package_url: ClassVar[str | None] = None
    package_sha: ClassVar[str | None] = None

    def step_init(
        self,
        flow: Any,
        graph: Any,
        step_name: str,
        decorators: Any,
        environment: Any,
        flow_datastore: Any,
        logger: Any,
    ) -> None:
        _ensure_conda_remote_command_aliases()

        self._step_name = step_name
        self.flow_datastore = flow_datastore
        self.environment = environment
        self.logger = logger
        # Always create fresh copies — defaults are shallow-copied by StepDecorator,
        # so mutating self.attributes["env"] would otherwise corrupt the class-level
        # defaults dict and leak state across decorator instances.
        self.attributes["env"] = dict(self.attributes.get("env") or {})
        self.attributes["secrets"] = list(self.attributes.get("secrets") or [])

        # Preserve Modal auth env vars through step-runtime transitions.
        for var in _MODAL_AUTH_ENV_VARS:
            val = os.environ.get(var) or _INITIAL_MODAL_AUTH_ENV.get(var)
            if val:
                self.attributes["env"].setdefault(var, val)

        # @modal requires a remote datastore — sandboxes can't reach the local FS.
        if flow_datastore.TYPE == "local":
            raise ModalException(
                "@modal requires a remote datastore (s3, azure, gs). "
                "Configure with: METAFLOW_DEFAULT_DATASTORE=s3\n"
                "See https://docs.metaflow.org/scaling/remote-tasks/introduction"
            )

    def runtime_init(self, flow: Any, graph: Any, package: Any, run_id: str) -> None:
        """Store flow-level state needed for code-package upload."""
        self.flow = flow
        self.graph = graph
        self.package = package
        self.run_id = run_id

    def runtime_task_created(
        self,
        task_datastore: Any,
        task_id: str,
        split_index: Any,
        input_paths: Any,
        is_cloned: bool,
        ubf_context: Any,
    ) -> None:
        """Upload the code package once per flow run."""
        if not is_cloned:
            self._save_package_once(self.flow_datastore, self.package)

    def runtime_step_cli(
        self,
        cli_args: Any,
        retry_count: int,
        max_user_code_retries: int,
        ubf_context: Any,
    ) -> None:
        """Redirect execution through ``modal step`` CLI command."""
        if os.environ.get("METAFLOW_MODAL_WORKLOAD"):
            return

        if retry_count <= max_user_code_retries:
            cli_args.commands = ["modal", "step"]
            cli_args.command_args.append(self.package_metadata)
            cli_args.command_args.append(self.package_sha)
            cli_args.command_args.append(self.package_url)

            _skip_keys = {"env", "secrets"}
            cli_args.command_options.update(
                {k: v for k, v in self.attributes.items() if k not in _skip_keys}
            )

            # Serialize user env vars as repeated --env-var KEY=VALUE
            user_env = dict(self.attributes.get("env") or {})
            for var in _MODAL_AUTH_ENV_VARS:
                val = os.environ.get(var) or _INITIAL_MODAL_AUTH_ENV.get(var)
                if val:
                    user_env.setdefault(var, val)
            if user_env:
                cli_args.command_options["env-var"] = [
                    f"{k}={v}" for k, v in user_env.items()
                ]

            # Serialize Modal secrets as repeated --secret <name>
            secrets = self.attributes.get("secrets") or []
            if secrets:
                cli_args.command_options["secret"] = list(secrets)

            cli_args.entrypoint[0] = sys.executable

    def task_pre_step(
        self,
        step_name: str,
        task_datastore: Any,
        metadata: Any,
        run_id: str,
        task_id: str,
        flow: Any,
        graph: Any,
        retry_count: int,
        max_user_code_retries: int,
        ubf_context: Any,
        inputs: Any,
    ) -> None:
        """Runs inside the Modal sandbox. Emit execution metadata."""
        self.metadata = metadata
        self.task_datastore = task_datastore

        if os.environ.get("METAFLOW_MODAL_WORKLOAD"):
            from metaflow.metadata_provider import MetaDatum

            meta = {
                "modal-sandbox-id": os.environ.get("METAFLOW_MODAL_SANDBOX_ID", ""),
            }
            entries = [
                MetaDatum(
                    field=k,
                    value=v,
                    type=k,
                    tags=[f"attempt_id:{retry_count}"],
                )
                for k, v in meta.items()
            ]
            metadata.register_metadata(run_id, step_name, task_id, entries)

    def task_finished(
        self,
        step_name: str,
        flow: Any,
        graph: Any,
        is_task_ok: bool,
        retry_count: int,
        max_retries: int,
    ) -> None:
        """Sync local metadata from datastore when running in Modal sandbox."""
        if (
            os.environ.get("METAFLOW_MODAL_WORKLOAD")
            and hasattr(self, "metadata")
            and self.metadata.TYPE == "local"
        ):
            from metaflow.metadata_provider.util import sync_local_metadata_to_datastore
            from metaflow.metaflow_config import DATASTORE_LOCAL_DIR

            sync_local_metadata_to_datastore(
                DATASTORE_LOCAL_DIR, self.task_datastore
            )

    @classmethod
    def _save_package_once(cls, flow_datastore: Any, package: Any) -> None:
        """Upload code package to remote datastore.

        Always stores on ``ModalDecorator`` so that if multiple steps use
        @modal, only one upload occurs per flow run.
        """
        if ModalDecorator.package_url is None:
            url, sha = flow_datastore.save_data([package.blob], len_hint=1)[0]
            ModalDecorator.package_url = url
            ModalDecorator.package_sha = sha
            ModalDecorator.package_metadata = package.package_metadata
