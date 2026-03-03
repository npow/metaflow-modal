"""Click CLI for Modal execution: ``python flow.py modal step ...``

Layer: CLI (same level as Metaflow Integration)
May only import from: .modal_executor, metaflow stdlib

Follows the exact same pattern as ``metaflow.plugins.aws.batch.batch_cli``.
Metaflow discovers this via the ``CLIS_DESC`` entry in ``plugins/__init__.py``.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from typing import Any

from metaflow import util
from metaflow._vendor import click
from metaflow.exception import METAFLOW_EXIT_DISALLOW_RETRY
from metaflow.metadata_provider.util import sync_local_metadata_from_datastore
from metaflow.metaflow_config import DATASTORE_LOCAL_DIR
from metaflow.unbounded_foreach import UBF_CONTROL
from metaflow.unbounded_foreach import UBF_TASK

from .modal_executor import ModalExecutor


def _replay_task_metadata_to_service(ctx: Any, run_id: str, step_name: str, task_id: str) -> None:
    """Replay locally-captured task metadata back to the metadata service.

    When running inside a Modal sandbox, Metaflow switches to local metadata
    (so it can write inside the sandbox without network access).  After the
    step completes, we need to push that metadata back to the service.
    """
    if ctx.obj.metadata.TYPE != "service":
        return

    from metaflow.plugins.metadata_providers.local import LocalMetadataProvider

    meta_dir = LocalMetadataProvider._get_metadir(
        ctx.obj.flow.name, run_id, step_name, task_id
    )
    if not meta_dir:
        return

    metadata_payload = []
    artifact_payload = []
    for path in glob.glob(os.path.join(meta_dir, "sysmeta_*.json")):
        with open(path) as f:
            metadata_payload.append(json.load(f))
    for path in glob.glob(os.path.join(meta_dir, "*_artifact_*.json")):
        with open(path) as f:
            artifact_payload.append(json.load(f))

    if not metadata_payload and not artifact_payload:
        return

    provider_cls = ctx.obj.metadata.__class__
    base_url = provider_cls._obj_path(ctx.obj.flow.name, run_id, step_name, task_id)

    ctx.obj.metadata.register_task_id(run_id, step_name, task_id)
    if metadata_payload:
        provider_cls._request(
            ctx.obj.monitor,
            base_url + "/metadata",
            "POST",
            metadata_payload,
        )
    if artifact_payload:
        provider_cls._request(
            ctx.obj.monitor,
            base_url + "/artifact",
            "POST",
            artifact_payload,
        )



@click.group()
def cli() -> None:
    pass


@cli.group(help="Commands related to Modal execution.")
def modal() -> None:
    pass


@modal.command(
    help="Execute a single task on Modal. "
    "This command runs the top-level step command inside a Modal sandbox. "
    "Typically you do not call this command directly; it is used internally by Metaflow."
)
@click.argument("step-name")
@click.argument("code-package-metadata")
@click.argument("code-package-sha")
@click.argument("code-package-url")
@click.option("--image", default=None, help="Docker image URL for the Modal sandbox.")
@click.option("--cpu", default=1.0, type=float, help="Number of CPUs.")
@click.option("--memory", default=1024, type=int, help="Memory in MB.")
@click.option("--gpu", default=None, help="GPU type (T4, A10G, A100, A100-80GB, H100, L4).")
@click.option("--timeout", default=600, type=int, help="Timeout in seconds.")
@click.option(
    "--env-var",
    "env_vars",
    multiple=True,
    default=None,
    help="User env vars from @modal(env={}). Format: KEY=VALUE, multiple allowed.",
)
@click.option(
    "--secret",
    "secrets",
    multiple=True,
    default=None,
    help="Modal secret name (from modal.Secret.from_name). Multiple allowed.",
)
@click.option("--run-id", help="Passed to the top-level 'step'.")
@click.option("--task-id", help="Passed to the top-level 'step'.")
@click.option("--input-paths", help="Passed to the top-level 'step'.")
@click.option("--split-index", help="Passed to the top-level 'step'.")
@click.option("--clone-path", help="Passed to the top-level 'step'.")
@click.option("--clone-run-id", help="Passed to the top-level 'step'.")
@click.option(
    "--tag", multiple=True, default=None, help="Passed to the top-level 'step'."
)
@click.option("--namespace", default=None, help="Passed to the top-level 'step'.")
@click.option("--retry-count", default=0, help="Passed to the top-level 'step'.")
@click.option(
    "--max-user-code-retries", default=0, help="Passed to the top-level 'step'."
)
@click.option(
    "--ubf-context",
    default=None,
    type=click.Choice(["none", UBF_CONTROL, UBF_TASK]),
)
@click.pass_context
def step(
    ctx: Any,
    step_name: str,
    code_package_metadata: str,
    code_package_sha: str,
    code_package_url: str,
    image: str | None = None,
    cpu: float = 1.0,
    memory: int = 1024,
    gpu: str | None = None,
    timeout: int = 600,
    env_vars: tuple[str, ...] | None = None,
    secrets: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> None:
    def echo(msg: str, stream: str = "stderr", **kw: Any) -> None:
        msg = util.to_unicode(msg)
        ctx.obj.echo_always(msg, err=(stream == "stderr"), **kw)

    executable = ctx.obj.environment.executable(step_name, None)
    entrypoint = f"{executable} -u {os.path.basename(sys.argv[0])}"

    top_params = dict(ctx.parent.parent.params)
    if ctx.obj.metadata.TYPE == "service":
        top_params["metadata"] = "local"
    top_args = " ".join(util.dict_to_cli_options(top_params))

    # Handle long input_paths by splitting into env vars.
    input_paths = kwargs.get("input_paths")
    split_vars: dict[str, str] | None = None
    if input_paths:
        max_size = 30 * 1024
        split_vars = {
            f"METAFLOW_INPUT_PATHS_{i // max_size}": input_paths[i : i + max_size]
            for i in range(0, len(input_paths), max_size)
        }
        kwargs["input_paths"] = "".join(f"${{{s}}}" for s in split_vars)

    step_args = " ".join(util.dict_to_cli_options(kwargs))
    step_cli = f"{entrypoint} {top_args} step {step_name} {step_args}"

    node = ctx.obj.graph[step_name]
    retry_count = kwargs.get("retry_count", 0)

    task_spec = {
        "flow_name": ctx.obj.flow.name,
        "step_name": step_name,
        "run_id": kwargs["run_id"],
        "task_id": kwargs["task_id"],
        "retry_count": str(retry_count),
    }

    # Build environment: start from @environment decorator vars, then user overrides.
    env: dict[str, str] = {"METAFLOW_FLOW_FILENAME": os.path.basename(sys.argv[0])}
    env_deco = [deco for deco in node.decorators if deco.name == "environment"]
    if env_deco:
        env.update(env_deco[0].attributes["vars"])
    if split_vars:
        env.update(split_vars)
    if env_vars:
        for item in list(env_vars):
            key, _, value = item.partition("=")
            if key:
                env[key] = value

    # Forward Modal auth vars into the executor process (needed by the SDK).
    for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_ENVIRONMENT"):
        value = env.get(key)
        if value:
            os.environ[key] = value

    def _sync_metadata() -> None:
        if ctx.obj.metadata.TYPE in ("local", "service"):
            sync_local_metadata_from_datastore(
                DATASTORE_LOCAL_DIR,
                ctx.obj.flow_datastore.get_task_datastore(
                    kwargs["run_id"], step_name, kwargs["task_id"]
                ),
            )
            if ctx.obj.metadata.TYPE == "service":
                try:
                    _replay_task_metadata_to_service(
                        ctx, kwargs["run_id"], step_name, kwargs["task_id"]
                    )
                except Exception as e:
                    echo(f"Modal metadata replay to service failed: {util.to_unicode(e)}")

    def _on_log(line: str, _stream: str) -> None:
        echo(line, stream="stderr")

    executor = ModalExecutor(ctx.obj.environment)
    try:
        executor.launch(
            step_name,
            step_cli,
            task_spec,
            code_package_metadata,
            code_package_sha,
            code_package_url,
            ctx.obj.flow_datastore.TYPE,
            image=image,
            cpu=cpu,
            memory=memory,
            gpu=gpu,
            timeout=timeout,
            env=env,
            secrets=list(secrets) if secrets else [],
            on_log=_on_log,
        )
    except Exception:
        traceback.print_exc()
        executor.cleanup()
        try:
            _sync_metadata()
        except Exception as e:
            echo(f"Modal metadata sync failed: {util.to_unicode(e)}", stream="stderr")
        sys.exit(METAFLOW_EXIT_DISALLOW_RETRY)

    try:
        executor.wait(echo=echo)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        executor.cleanup()
        sys.exit(METAFLOW_EXIT_DISALLOW_RETRY)
    finally:
        try:
            _sync_metadata()
        except Exception as e:
            echo(f"Modal metadata sync failed: {util.to_unicode(e)}", stream="stderr")
