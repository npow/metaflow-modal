# Metaflow plugin registration.
# Metaflow discovers these descriptors at import time via the
# metaflow_extensions namespace package convention.
#
# Layer: Plugin Registration (top-level entry point)
# May only import from: .modal_decorator, .modal_cli

STEP_DECORATORS_DESC = [
    ("modal", ".modal_decorator.ModalDecorator"),
]

CLIS_DESC = [
    ("modal", ".modal_cli.cli"),
]
