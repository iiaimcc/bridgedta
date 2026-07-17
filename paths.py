"""Path helpers for the BridgeDTA repository.

Resolves the project root as the directory containing this file, regardless of
the current working directory of the process that imports it. All scripts should
call ``get_repo_root()`` instead of hard-coding absolute paths or appending to
``sys.path``.

Example
-------
    from paths import get_repo_root
    repo_root = get_repo_root()
    data_dir = repo_root / "data"
    feature_dir = repo_root / "models" / "Bridge_DTA" / "feature_kpgt"
"""
from __future__ import annotations

import os
from pathlib import Path


def get_repo_root() -> Path:
    """Return the absolute path to the BridgeDTA repository root.

    Resolved from the location of this file, so callers may run from any cwd.
    """
    return Path(__file__).resolve().parent


def repo_path(*parts: str) -> Path:
    """Join ``parts`` against the repository root and return a ``Path``."""
    return get_repo_root().joinpath(*parts)


def data_dir() -> Path:
    """Path to ``<repo>/data``."""
    return repo_path("data")


def results_dir() -> Path:
    """Path to ``<repo>/results`` (created on demand by the trainer)."""
    return repo_path("results")


def feature_dir(name: str) -> Path:
    """Path to a pre-computed feature directory under ``models/Bridge_DTA``.

    ``name`` should be ``feature_kpgt`` or ``feature_esm``. The directory is not
    required to exist here; callers that need features should check existence
    and point the user to the README download step.
    """
    return repo_path("models", "Bridge_DTA", name)


def setup_imports() -> None:
    """Insert the repository root at the front of ``sys.path``.

    Convenience for entry-point scripts: call once at the top and the package
    modules (``trainer``, ``datahelper.*``, ``models.*``) become importable no
    matter the cwd.
    """
    import sys

    root = str(get_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def apply_training_cli_overrides(args, *, keys=("dataset", "num_epochs", "batch_size", "seed", "log_steps", "cold_start")):
    """Apply ``--key value`` overrides from ``sys.argv`` onto a training args object.

    Designed for the ``TrainingArguments`` classes used across ``experiments/``
    and ``exp_fold/``. Each requested key, if present in ``args`` and given on
    the command line as ``--dataset davis`` etc., is cast to the existing
    attribute's type and assigned back. Unknown flags are ignored, so scripts
    can still be invoked with extra arguments without breaking.

    Example::

        class TrainingArguments:
            def __init__(self):
                self.dataset = "kiba"
                self.num_epochs = 500
                ...
                apply_training_cli_overrides(self)
    """
    import sys

    flag_values = {}
    args_iter = iter(sys.argv[1:])
    for token in args_iter:
        if token.startswith("--") and "=" in token:
            name, value = token[2:].split("=", 1)
            flag_values[name] = value
        elif token.startswith("--"):
            name = token[2:]
            if name in keys:
                try:
                    flag_values[name] = next(args_iter)
                except StopIteration:
                    pass
    for key in keys:
        if key not in flag_values:
            continue
        if not hasattr(args, key):
            continue
        current = getattr(args, key)
        try:
            if isinstance(current, bool):
                setattr(args, key, flag_values[key].lower() in ("1", "true", "yes", "on"))
            else:
                setattr(args, key, type(current)(flag_values[key]))
        except (TypeError, ValueError):
            raise ValueError(
                f"Could not cast --{key}={flag_values[key]!r} to {type(current).__name__}"
            )
