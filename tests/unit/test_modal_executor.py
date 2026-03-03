"""Unit tests for ModalExecutor.

Uses AST-based source inspection and lightweight mocking to avoid
importing metaflow or modal directly (which may not be installed in CI).
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

EXECUTOR_FILE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "metaflow_extensions"
    / "modal"
    / "plugins"
    / "modal_executor.py"
)


# ---------------------------------------------------------------------------
# Source-based structure tests (no imports needed)
# ---------------------------------------------------------------------------


class TestExecutorStructure:
    """Verify the executor module has expected classes and methods."""

    def test_file_exists(self) -> None:
        assert EXECUTOR_FILE.exists()

    def test_has_modal_executor_class(self) -> None:
        tree = ast.parse(EXECUTOR_FILE.read_text())
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert "ModalExecutor" in class_names

    def test_has_modal_exception_class(self) -> None:
        tree = ast.parse(EXECUTOR_FILE.read_text())
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert "ModalException" in class_names

    @pytest.fixture()
    def executor_methods(self) -> set[str]:
        tree = ast.parse(EXECUTOR_FILE.read_text())
        methods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModalExecutor":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
        return methods

    def test_has_command_method(self, executor_methods: set[str]) -> None:
        assert "_command" in executor_methods

    def test_has_build_env_method(self, executor_methods: set[str]) -> None:
        assert "_build_env" in executor_methods

    def test_has_build_image_method(self, executor_methods: set[str]) -> None:
        assert "_build_image" in executor_methods

    def test_has_resolve_gpu_method(self, executor_methods: set[str]) -> None:
        assert "_resolve_gpu" in executor_methods

    def test_has_launch_method(self, executor_methods: set[str]) -> None:
        assert "launch" in executor_methods

    def test_has_wait_method(self, executor_methods: set[str]) -> None:
        assert "wait" in executor_methods

    def test_has_cleanup_method(self, executor_methods: set[str]) -> None:
        assert "cleanup" in executor_methods


class TestExecutorCommandBuilding:
    """Verify the _command method builds proper bash commands."""

    def test_uses_mflog_env_vars(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "export_mflog_env_vars" in source

    def test_uses_bash_capture_logs(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "bash_capture_logs" in source

    def test_uses_bash_save_logs(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "BASH_SAVE_LOGS" in source

    def test_creates_logs_dir(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "mkdir -p" in source

    def test_preserves_exit_code(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "exit $c" in source


class TestExecutorEnvVars:
    """Verify environment variable assembly."""

    def test_includes_code_package_vars(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "METAFLOW_CODE_METADATA" in source
        assert "METAFLOW_CODE_SHA" in source
        assert "METAFLOW_CODE_URL" in source
        assert "METAFLOW_CODE_DS" in source

    def test_includes_modal_workload_marker(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "METAFLOW_MODAL_WORKLOAD" in source

    def test_includes_credential_forwarding(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "AWS_ACCESS_KEY_ID" in source
        assert "AWS_SECRET_ACCESS_KEY" in source
        assert "AWS_SESSION_TOKEN" in source
        assert "MODAL_TOKEN_ID" in source
        assert "MODAL_TOKEN_SECRET" in source

    def test_includes_datastore_vars(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "METAFLOW_DEFAULT_DATASTORE" in source
        assert "METAFLOW_USER" in source
        assert "METAFLOW_CONDA" in source

    def test_forces_local_metadata_when_service(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert 'proxy_service_metadata = DEFAULT_METADATA == "service"' in source
        assert '"METAFLOW_DEFAULT_METADATA": (' in source

    def test_forwards_metaflow_s3_config(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "METAFLOW_S3" in source
        assert "METAFLOW_DATASTORE_SYSROOT_" in source


class TestExecutorImageBuilding:
    """Verify Modal Image construction logic."""

    def test_none_image_uses_debian_slim(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "debian_slim" in source

    def test_custom_image_uses_from_registry(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "from_registry" in source

    def test_install_hint_has_pip_command(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "_INSTALL_HINT" in source
        assert "pip install" in source
        assert "MODAL_TOKEN_ID" in source


class TestExecutorGpuResolution:
    """Verify GPU string-to-Modal-config conversion."""

    def _get_resolve_gpu(self) -> Any:
        """Import _resolve_gpu with a mocked modal module."""
        modal_mock = MagicMock()
        modal_mock.gpu.T4 = MagicMock(return_value="t4-config")
        modal_mock.gpu.A10G = MagicMock(return_value="a10g-config")
        modal_mock.gpu.A100 = MagicMock(return_value="a100-config")
        modal_mock.gpu.H100 = MagicMock(return_value="h100-config")
        modal_mock.gpu.L4 = MagicMock(return_value="l4-config")

        with patch.dict("sys.modules", {"modal": modal_mock}):
            # Force reimport of the executor with mock in place.
            import importlib
            import sys as _sys

            # Remove cached module if present
            for key in list(_sys.modules.keys()):
                if "modal_executor" in key:
                    del _sys.modules[key]

            from metaflow_extensions.modal.plugins.modal_executor import ModalExecutor

            return ModalExecutor._resolve_gpu, modal_mock

    def test_none_returns_none(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "if not gpu:" in source
        assert "return None" in source

    def test_gpu_map_has_common_types(self) -> None:
        source = EXECUTOR_FILE.read_text()
        for gpu_type in ("T4", "A10G", "A100", "H100", "L4"):
            assert gpu_type in source

    def test_count_suffix_supported(self) -> None:
        source = EXECUTOR_FILE.read_text()
        # COUNT suffix requires split on ":"
        assert '":"' in source or '":", 1' in source or "split(\":\", 1)" in source


class TestExecutorLaunch:
    """Verify the launch method uses Modal Sandbox correctly."""

    def test_uses_sandbox_create(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "modal.Sandbox.create" in source

    def test_passes_image(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "image=modal_image" in source

    def test_passes_timeout(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "timeout=timeout" in source

    def test_passes_environment_variables(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "env=sandbox_env" in source

    def test_passes_cpu_as_float(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "cpu=float(cpu)" in source

    def test_passes_memory(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "memory=memory" in source

    def test_passes_gpu(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "gpu=modal_gpu" in source

    def test_passes_secrets(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "modal.Secret.from_name" in source
        assert "secrets=modal_secrets" in source


class TestExecutorWait:
    """Verify the wait method handles streaming and exit codes."""

    def test_uses_threading_for_streaming(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "threading.Thread" in source

    def test_calls_sandbox_wait(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "sandbox.wait()" in source

    def test_reads_returncode(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "sandbox.returncode" in source

    def test_calls_cleanup_after_wait(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "self.cleanup()" in source

    def test_exits_on_nonzero(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "sys.exit(exit_code)" in source

    def test_streams_stdout_and_stderr(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "sandbox.stdout" in source
        assert "sandbox.stderr" in source


class TestExecutorCleanup:
    """Verify the cleanup method terminates the sandbox."""

    def test_calls_terminate(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "sandbox.terminate()" in source or ".terminate()" in source

    def test_suppresses_cleanup_exceptions(self) -> None:
        source = EXECUTOR_FILE.read_text()
        assert "suppress(Exception)" in source


# ---------------------------------------------------------------------------
# Behavioural tests (require mocked modal + metaflow)
# ---------------------------------------------------------------------------

from typing import Any  # noqa: E402


def _make_mock_modal() -> MagicMock:
    """Build a minimal modal mock sufficient for ModalExecutor tests."""
    modal_mock = MagicMock()

    # Image API
    debian_slim_mock = MagicMock()
    modal_mock.Image.debian_slim.return_value = debian_slim_mock
    from_reg_mock = MagicMock()
    modal_mock.Image.from_registry.return_value = from_reg_mock

    # GPU API
    modal_mock.gpu.T4 = MagicMock(return_value="t4-config")
    modal_mock.gpu.A10G = MagicMock(return_value="a10g-config")
    modal_mock.gpu.A100 = MagicMock(return_value="a100-config")
    modal_mock.gpu.H100 = MagicMock(return_value="h100-config")
    modal_mock.gpu.L4 = MagicMock(return_value="l4-config")

    # Secret API
    modal_mock.Secret.from_name = MagicMock(side_effect=lambda name: f"secret:{name}")

    # Sandbox API
    sandbox_mock = MagicMock()
    sandbox_mock.returncode = 0
    sandbox_mock.stdout.__iter__ = MagicMock(return_value=iter(["line1\n", "line2\n"]))
    sandbox_mock.stderr.__iter__ = MagicMock(return_value=iter([]))
    modal_mock.Sandbox.create.return_value = sandbox_mock

    return modal_mock


def _make_mock_environment() -> MagicMock:
    env = MagicMock()
    env.executable.return_value = "python"
    env.get_package_commands.return_value = ["echo setup"]
    env.bootstrap_commands.return_value = ["echo bootstrap"]
    return env


def _make_mock_metaflow() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Return mocks for metaflow modules the executor imports."""
    util_mock = MagicMock()
    util_mock.get_username.return_value = "testuser"

    mflog_mock = MagicMock()
    mflog_mock.export_mflog_env_vars.return_value = "export MFLOG=1"
    mflog_mock.bash_capture_logs.side_effect = lambda cmd: cmd
    mflog_mock.BASH_SAVE_LOGS = "save_logs"

    config_mock = MagicMock()
    config_mock.DEFAULT_METADATA = "local"
    config_mock.SERVICE_INTERNAL_URL = None
    config_mock.DATASTORE_LOCAL_DIR = "/tmp/mf"

    config_funcs_mock = MagicMock()
    config_funcs_mock.config_values.return_value = []

    return util_mock, mflog_mock, config_mock, config_funcs_mock


@pytest.fixture()
def mocked_executor():
    """Return a ModalExecutor instance with all external deps mocked."""
    modal_mock = _make_mock_modal()
    util_mock, mflog_mock, config_mock, config_funcs_mock = _make_mock_metaflow()

    modules = {
        "modal": modal_mock,
        "metaflow": MagicMock(),
        "metaflow.util": util_mock,
        "metaflow.exception": MagicMock(),
        "metaflow.metaflow_config": config_mock,
        "metaflow.mflog": mflog_mock,
        "metaflow.metaflow_config_funcs": config_funcs_mock,
    }

    with patch.dict("sys.modules", modules):
        import importlib
        import sys as _sys

        for key in list(_sys.modules.keys()):
            if "modal_executor" in key:
                del _sys.modules[key]

        from metaflow_extensions.modal.plugins.modal_executor import ModalExecutor

        env = _make_mock_environment()
        executor = ModalExecutor(env)

    return executor, modal_mock


class TestExecutorBehaviour:
    """Behavioural tests for ModalExecutor with mocked dependencies."""

    def test_launch_calls_sandbox_create(self, mocked_executor: Any) -> None:
        executor, modal_mock = mocked_executor
        with patch.dict("sys.modules", {"modal": modal_mock}):
            executor.launch(
                step_name="my_step",
                step_cli="python flow.py step my_step",
                task_spec={
                    "flow_name": "MyFlow",
                    "step_name": "my_step",
                    "run_id": "1",
                    "task_id": "1",
                    "retry_count": "0",
                },
                code_package_metadata="meta",
                code_package_sha="sha",
                code_package_url="s3://bucket/code.tar",
                datastore_type="s3",
                image=None,
                cpu=2.0,
                memory=4096,
                gpu=None,
                timeout=300,
                env={"MY_VAR": "hello"},
                secrets=[],
            )
        modal_mock.Sandbox.create.assert_called_once()
        call_kwargs = modal_mock.Sandbox.create.call_args
        assert call_kwargs.kwargs["cpu"] == 2.0
        assert call_kwargs.kwargs["memory"] == 4096
        assert call_kwargs.kwargs["timeout"] == 300

    def test_launch_with_gpu_resolves_modal_gpu(self, mocked_executor: Any) -> None:
        executor, modal_mock = mocked_executor
        with patch.dict("sys.modules", {"modal": modal_mock}):
            executor.launch(
                step_name="gpu_step",
                step_cli="python flow.py step gpu_step",
                task_spec={
                    "flow_name": "MyFlow",
                    "step_name": "gpu_step",
                    "run_id": "1",
                    "task_id": "2",
                    "retry_count": "0",
                },
                code_package_metadata="meta",
                code_package_sha="sha",
                code_package_url="s3://bucket/code.tar",
                datastore_type="s3",
                gpu="A10G",
            )
        call_kwargs = modal_mock.Sandbox.create.call_args
        assert call_kwargs.kwargs["gpu"] == "a10g-config"

    def test_launch_with_secrets(self, mocked_executor: Any) -> None:
        executor, modal_mock = mocked_executor
        with patch.dict("sys.modules", {"modal": modal_mock}):
            executor.launch(
                step_name="my_step",
                step_cli="python flow.py step my_step",
                task_spec={
                    "flow_name": "MyFlow",
                    "step_name": "my_step",
                    "run_id": "1",
                    "task_id": "3",
                    "retry_count": "0",
                },
                code_package_metadata="meta",
                code_package_sha="sha",
                code_package_url="s3://bucket/code.tar",
                datastore_type="s3",
                secrets=["my-aws-creds"],
            )
        modal_mock.Secret.from_name.assert_called_with("my-aws-creds")
        call_kwargs = modal_mock.Sandbox.create.call_args
        assert "secret:my-aws-creds" in call_kwargs.kwargs["secrets"]

    def test_cleanup_terminates_sandbox(self, mocked_executor: Any) -> None:
        executor, modal_mock = mocked_executor
        sandbox_mock = modal_mock.Sandbox.create.return_value
        executor._sandbox = sandbox_mock
        executor.cleanup()
        sandbox_mock.terminate.assert_called_once()
        assert executor._sandbox is None

    def test_cleanup_noop_when_no_sandbox(self, mocked_executor: Any) -> None:
        executor, _ = mocked_executor
        executor._sandbox = None
        executor.cleanup()  # must not raise
