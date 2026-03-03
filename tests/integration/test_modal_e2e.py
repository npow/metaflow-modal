"""End-to-end integration tests for metaflow-modal.

Requirements:
    - Modal credentials configured (modal token new, or MODAL_TOKEN_ID + MODAL_TOKEN_SECRET)
    - Remote datastore configured (METAFLOW_DEFAULT_DATASTORE=s3)

Run:
    pytest tests/integration/ -m integration -v
    pytest tests/integration/test_modal_e2e.py::TestModalE2E::test_simple_artifact_roundtrip -v
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _skip_if_no_modal() -> None:
    """Skip the test if Modal credentials are not available."""
    try:
        import modal  # noqa: F401
    except ImportError:
        pytest.skip("modal SDK not installed")

    # modal.config.Config checks token from ~/.modal.toml or env vars
    token_id = os.environ.get("MODAL_TOKEN_ID")
    token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    config_file = os.path.expanduser("~/.modal.toml")

    if not ((token_id and token_secret) or os.path.exists(config_file)):
        pytest.skip(
            "No Modal credentials found. Run `modal token new` or set "
            "MODAL_TOKEN_ID + MODAL_TOKEN_SECRET."
        )


def _run_flow(
    flow_src: str, tmp_path: any, extra_args: list[str] | None = None
) -> subprocess.CompletedProcess:
    """Write a flow to a temp file, run it, and return the result."""
    flow_file = tmp_path / "flow.py"
    flow_file.write_text(flow_src)
    cmd = [sys.executable, str(flow_file), "run"] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


@pytest.mark.integration
class TestModalE2E:
    """Full end-to-end tests that spin up real Modal sandboxes."""

    def test_simple_artifact_roundtrip(self, tmp_path) -> None:
        """A minimal flow that writes an artifact in Modal and reads it back."""
        _skip_if_no_modal()

        result = _run_flow(
            textwrap.dedent("""\
                from metaflow import FlowSpec, step
                from metaflow_extensions.modal.plugins.modal_decorator import ModalDecorator

                # Register decorator so Metaflow can find it when loaded standalone.
                try:
                    from metaflow import modal  # noqa: F401
                except Exception:
                    pass

                class ModalArtifactFlow(FlowSpec):
                    @step
                    def start(self):
                        self.message = "hello from Modal"
                        self.next(self.compute)

                    @modal(cpu=1, memory=512, timeout=120)
                    @step
                    def compute(self):
                        import platform
                        self.platform = platform.system()
                        assert self.message == "hello from Modal"
                        self.result = self.message.upper()
                        self.next(self.end)

                    @step
                    def end(self):
                        assert self.result == "HELLO FROM MODAL"
                        assert self.platform == "Linux"
                        print("Success:", self.result)

                if __name__ == "__main__":
                    ModalArtifactFlow()
            """),
            tmp_path,
        )

        assert result.returncode == 0, (
            f"Flow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        success = "Success: HELLO FROM MODAL"
        assert success in result.stdout or success in result.stderr

    def test_gpu_metadata(self, tmp_path) -> None:
        """Flow that requests a T4 GPU and verifies nvidia-smi is available."""
        _skip_if_no_modal()

        result = _run_flow(
            textwrap.dedent("""\
                from metaflow import FlowSpec, step

                class ModalGpuFlow(FlowSpec):
                    @step
                    def start(self):
                        self.next(self.gpu_step)

                    @modal(cpu=1, memory=1024, gpu="T4", timeout=180)
                    @step
                    def gpu_step(self):
                        import subprocess
                        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                        assert r.returncode == 0, f"nvidia-smi failed: {r.stderr}"
                        assert "GPU" in r.stdout, f"No GPU found: {r.stdout}"
                        self.gpu_info = r.stdout.strip()
                        self.next(self.end)

                    @step
                    def end(self):
                        print("GPU info:", self.gpu_info)

                if __name__ == "__main__":
                    ModalGpuFlow()
            """),
            tmp_path,
        )

        assert result.returncode == 0, (
            f"GPU flow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_custom_image(self, tmp_path) -> None:
        """Flow that uses a custom Docker image."""
        _skip_if_no_modal()

        result = _run_flow(
            textwrap.dedent("""\
                from metaflow import FlowSpec, step

                class ModalCustomImageFlow(FlowSpec):
                    @step
                    def start(self):
                        self.next(self.compute)

                    @modal(image="python:3.12-slim", cpu=1, memory=512, timeout=120)
                    @step
                    def compute(self):
                        import sys
                        self.py_version = sys.version
                        assert sys.version_info >= (3, 12), f"Expected 3.12+, got {sys.version}"
                        self.next(self.end)

                    @step
                    def end(self):
                        print("Python version:", self.py_version)

                if __name__ == "__main__":
                    ModalCustomImageFlow()
            """),
            tmp_path,
        )

        assert result.returncode == 0, (
            f"Custom image flow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_env_vars_forwarded(self, tmp_path) -> None:
        """Flow that verifies custom env vars reach the sandbox."""
        _skip_if_no_modal()

        env = {**os.environ, "MY_TEST_VAR": "modal_e2e_value"}

        flow_file = tmp_path / "flow.py"
        flow_file.write_text(
            textwrap.dedent("""\
                from metaflow import FlowSpec, step

                class ModalEnvFlow(FlowSpec):
                    @step
                    def start(self):
                        self.next(self.compute)

                    @modal(cpu=1, memory=512, timeout=120,
                           env={"MY_TEST_VAR": "modal_e2e_value"})
                    @step
                    def compute(self):
                        import os
                        val = os.environ.get("MY_TEST_VAR")
                        assert val == "modal_e2e_value", f"Expected modal_e2e_value, got {val!r}"
                        self.env_val = val
                        self.next(self.end)

                    @step
                    def end(self):
                        assert self.env_val == "modal_e2e_value"
                        print("env var ok:", self.env_val)

                if __name__ == "__main__":
                    ModalEnvFlow()
            """)
        )

        result = subprocess.run(
            [sys.executable, str(flow_file), "run"],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )

        assert result.returncode == 0, (
            f"Env var flow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_foreach(self, tmp_path) -> None:
        """Flow with a foreach step — verifies parallel task fan-out works."""
        _skip_if_no_modal()

        result = _run_flow(
            textwrap.dedent("""\
                from metaflow import FlowSpec, step

                class ModalForeachFlow(FlowSpec):
                    @step
                    def start(self):
                        self.items = [1, 2, 3]
                        self.next(self.process, foreach="items")

                    @modal(cpu=1, memory=512, timeout=120)
                    @step
                    def process(self):
                        self.squared = self.input ** 2
                        self.next(self.join)

                    @step
                    def join(self, inputs):
                        self.results = sorted(inp.squared for inp in inputs)
                        self.next(self.end)

                    @step
                    def end(self):
                        assert self.results == [1, 4, 9], f"Got {self.results}"
                        print("foreach ok:", self.results)

                if __name__ == "__main__":
                    ModalForeachFlow()
            """),
            tmp_path,
        )

        assert result.returncode == 0, (
            f"Foreach flow failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
