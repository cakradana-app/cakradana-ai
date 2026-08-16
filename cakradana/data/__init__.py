"""Synthetic data generation and its acceptance checks."""

from cakradana.data.acceptance import (
    AcceptanceError,
    MIN_RECALL,
    TypologyCheck,
    assert_acceptable,
    check,
)
from cakradana.data.generator import (
    ALL_TYPOLOGIES,
    GENERATOR_VERSION,
    GeneratorConfig,
    SyntheticDataset,
    generate,
)

__all__ = [
    "ALL_TYPOLOGIES",
    "AcceptanceError",
    "GENERATOR_VERSION",
    "GeneratorConfig",
    "MIN_RECALL",
    "SyntheticDataset",
    "TypologyCheck",
    "assert_acceptable",
    "check",
    "generate",
]
