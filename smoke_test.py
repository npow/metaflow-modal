#!/usr/bin/env python3
"""Standalone smoke test for metaflow-modal.

Run this directly to verify the extension works end-to-end:

    modal token new                 # if not yet configured
    python smoke_test.py

Requirements:
    - Modal credentials (modal token new OR MODAL_TOKEN_ID + MODAL_TOKEN_SECRET)
    - Remote S3 datastore (METAFLOW_DEFAULT_DATASTORE=s3 is already set on Netflix hosts)
"""

from metaflow import FlowSpec, step
from metaflow import modal


class ModalSmokeFlow(FlowSpec):
    @step
    def start(self):
        self.message = "hello from local"
        self.next(self.remote)

    @modal(cpu=1, memory=512, timeout=120)
    @step
    def remote(self):
        import platform
        import sys

        print(f"Running on: {platform.system()} {platform.machine()}")
        print(f"Python: {sys.version}")

        assert platform.system() == "Linux", "Should be Linux inside Modal"
        self.platform = platform.system()
        self.py_version = sys.version_info[:2]
        self.result = self.message.upper()
        self.next(self.end)

    @step
    def end(self):
        assert self.result == "HELLO FROM LOCAL"
        assert self.platform == "Linux"
        print(f"Smoke test passed! result={self.result}, py={self.py_version}")


if __name__ == "__main__":
    ModalSmokeFlow()
