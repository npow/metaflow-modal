"""Structural tests that enforce architectural invariants.

These tests require no credentials or external services. They inspect the
source code and module structure to catch layer violations, missing
implementations, and documentation gaps mechanically.

Run: pytest tests/structural/ -m structural
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PLUGINS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "metaflow_extensions"
    / "modal"
    / "plugins"
)

_EXPECTED_PLUGIN_FILES = [
    "__init__.py",
    "modal_decorator.py",
    "modal_cli.py",
    "modal_executor.py",
]


def _parse_imports(filepath: Path) -> set[str]:
    """Return all imported module names from a Python file."""
    tree = ast.parse(filepath.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestFileExistence:
    """All expected plugin files must exist."""

    @pytest.mark.parametrize("filename", _EXPECTED_PLUGIN_FILES)
    def test_plugin_file_exists(self, filename: str) -> None:
        assert (PLUGINS_DIR / filename).exists(), (
            f"Expected plugin file {filename} not found in {PLUGINS_DIR}."
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestPluginRegistration:
    """CLIS_DESC and STEP_DECORATORS_DESC must be well-formed."""

    def test_clis_desc_exists(self) -> None:
        tree = ast.parse((PLUGINS_DIR / "__init__.py").read_text())
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        source = (PLUGINS_DIR / "__init__.py").read_text()
        assert "CLIS_DESC" in source

    def test_step_decorators_desc_has_modal(self) -> None:
        source = (PLUGINS_DIR / "__init__.py").read_text()
        assert "STEP_DECORATORS_DESC" in source
        assert "modal" in source
        assert "ModalDecorator" in source

    def test_clis_desc_has_modal(self) -> None:
        source = (PLUGINS_DIR / "__init__.py").read_text()
        assert "CLIS_DESC" in source
        assert "modal" in source
        assert "modal_cli" in source


# ---------------------------------------------------------------------------
# Layer boundary enforcement
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestLayerBoundaries:
    """Lower layers must not import from higher layers."""

    def test_executor_does_not_import_decorator(self) -> None:
        imports = _parse_imports(PLUGINS_DIR / "modal_executor.py")
        assert "metaflow_extensions.modal.plugins.modal_decorator" not in imports, (
            "modal_executor.py must not import from modal_decorator.py."
        )

    def test_executor_does_not_import_cli(self) -> None:
        imports = _parse_imports(PLUGINS_DIR / "modal_executor.py")
        assert "metaflow_extensions.modal.plugins.modal_cli" not in imports, (
            "modal_executor.py must not import from modal_cli.py."
        )

    def test_decorator_does_not_import_executor(self) -> None:
        imports = _parse_imports(PLUGINS_DIR / "modal_decorator.py")
        assert "metaflow_extensions.modal.plugins.modal_executor" not in imports, (
            "modal_decorator.py must not import from modal_executor.py "
            "(decorator only redirects via CLI, never calls executor directly)."
        )


# ---------------------------------------------------------------------------
# Lazy imports (no top-level SDK imports)
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestLazyImports:
    """The modal SDK must not be imported at module top level."""

    def _has_toplevel_import(self, filepath: Path, pkg: str) -> bool:
        """Return True if *pkg* is imported at the module's top level (not inside a def/class)."""
        tree = ast.parse(filepath.read_text())
        for node in tree.body:  # only top-level statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == pkg or alias.name.startswith(pkg + "."):
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and (
                    node.module == pkg or node.module.startswith(pkg + ".")
                ):
                    return True
        return False

    def test_executor_no_toplevel_modal_import(self) -> None:
        assert not self._has_toplevel_import(PLUGINS_DIR / "modal_executor.py", "modal"), (
            "modal_executor.py imports 'modal' at the top level. "
            "Import inside methods so that importing the module never raises "
            "ImportError when modal is not installed."
        )

    def test_decorator_no_toplevel_modal_import(self) -> None:
        assert not self._has_toplevel_import(PLUGINS_DIR / "modal_decorator.py", "modal"), (
            "modal_decorator.py imports 'modal' at the top level."
        )

    def test_cli_no_toplevel_modal_import(self) -> None:
        assert not self._has_toplevel_import(PLUGINS_DIR / "modal_cli.py", "modal"), (
            "modal_cli.py imports 'modal' at the top level."
        )


# ---------------------------------------------------------------------------
# Docstrings
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestDocstrings:
    """Every plugin file must declare its layer in the module docstring."""

    @pytest.mark.parametrize(
        "filename",
        ["modal_decorator.py", "modal_cli.py", "modal_executor.py"],
    )
    def test_layer_declared(self, filename: str) -> None:
        tree = ast.parse((PLUGINS_DIR / filename).read_text())
        docstring = ast.get_docstring(tree)
        assert docstring is not None, (
            f"{filename} is missing a module docstring."
        )
        assert "Layer:" in docstring, (
            f"{filename} module docstring must contain 'Layer:' declaration."
        )


# ---------------------------------------------------------------------------
# Error messages
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestErrorMessages:
    """The executor must have an _INSTALL_HINT with remediation instructions."""

    def test_install_hint_exists(self) -> None:
        source = (PLUGINS_DIR / "modal_executor.py").read_text()
        assert "_INSTALL_HINT" in source

    def test_install_hint_has_pip_command(self) -> None:
        source = (PLUGINS_DIR / "modal_executor.py").read_text()
        assert "pip install" in source

    def test_install_hint_has_token_instruction(self) -> None:
        source = (PLUGINS_DIR / "modal_executor.py").read_text()
        assert "MODAL_TOKEN_ID" in source

    def test_decorator_validates_datastore(self) -> None:
        source = (PLUGINS_DIR / "modal_decorator.py").read_text()
        assert "local" in source
        assert "remote datastore" in source


# ---------------------------------------------------------------------------
# Decorator contract
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestDecoratorContract:
    """The ModalDecorator must implement the expected lifecycle hooks."""

    @pytest.fixture()
    def decorator_methods(self) -> set[str]:
        tree = ast.parse((PLUGINS_DIR / "modal_decorator.py").read_text())
        methods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModalDecorator":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(item.name)
        return methods

    def test_has_step_init(self, decorator_methods: set[str]) -> None:
        assert "step_init" in decorator_methods

    def test_has_runtime_init(self, decorator_methods: set[str]) -> None:
        assert "runtime_init" in decorator_methods

    def test_has_runtime_task_created(self, decorator_methods: set[str]) -> None:
        assert "runtime_task_created" in decorator_methods

    def test_has_runtime_step_cli(self, decorator_methods: set[str]) -> None:
        assert "runtime_step_cli" in decorator_methods

    def test_has_task_pre_step(self, decorator_methods: set[str]) -> None:
        assert "task_pre_step" in decorator_methods

    def test_has_task_finished(self, decorator_methods: set[str]) -> None:
        assert "task_finished" in decorator_methods

    def test_redirects_to_modal_cli(self) -> None:
        source = (PLUGINS_DIR / "modal_decorator.py").read_text()
        assert '["modal", "step"]' in source

    def test_sets_linux64_target_platform(self) -> None:
        source = (PLUGINS_DIR / "modal_decorator.py").read_text()
        assert "linux-64" in source

    def test_class_level_package_state(self) -> None:
        source = (PLUGINS_DIR / "modal_decorator.py").read_text()
        assert "package_url" in source
        assert "package_sha" in source
        assert "package_metadata" in source

    def test_skips_redirect_when_inside_sandbox(self) -> None:
        source = (PLUGINS_DIR / "modal_decorator.py").read_text()
        assert "METAFLOW_MODAL_WORKLOAD" in source


# ---------------------------------------------------------------------------
# CLI contract
# ---------------------------------------------------------------------------


@pytest.mark.structural
class TestCLIContract:
    """The modal CLI must expose the expected commands and options."""

    def test_has_step_command(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "def step(" in source

    def test_step_accepts_gpu_option(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "--gpu" in source

    def test_step_accepts_cpu_option(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "--cpu" in source

    def test_step_accepts_memory_option(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "--memory" in source

    def test_step_accepts_secrets_option(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "--secret" in source

    def test_step_accepts_env_var_option(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "--env-var" in source

    def test_syncs_metadata_on_exit(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "sync_local_metadata_from_datastore" in source

    def test_disallows_retry_on_exception(self) -> None:
        source = (PLUGINS_DIR / "modal_cli.py").read_text()
        assert "METAFLOW_EXIT_DISALLOW_RETRY" in source
