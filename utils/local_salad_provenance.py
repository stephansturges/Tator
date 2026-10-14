"""Shared provenance and isolated framework RNG for Tator-local SALAD training.

This qualifies new runs. A legacy checkpoint's metadata seed is never upgraded
to proof of deterministic initialization or of image-content identity.
"""
from contextlib import contextmanager
import hashlib
import importlib.metadata
import random
import marshal
import threading
from pathlib import Path

import numpy as np

TRAINING_IDENTITY_VERSION = "tator-local-salad-qualified-v2"
TRAINING_RNG_LOCK = threading.RLock()


def sha256_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def training_implementation_identity():
    root = Path(__file__).resolve().parent
    from utils.local_salad import LocalSALADHead, _PrivateDropout, _head_linear
    from utils.local_salad_mlx import MLXLocalSALADHead, _PrivateMLXDropout, _PrivateMLXLinear
    loaded_codes = {label: hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest() for label, fn in {
        "torch_linear_init": _head_linear, "mlx_linear_init": _PrivateMLXLinear.__init__, "mlx_linear_forward": _PrivateMLXLinear.__call__,
        "torch_init": LocalSALADHead.__init__, "torch_forward": LocalSALADHead.forward,
        "torch_dropout": _PrivateDropout.forward, "mlx_init": MLXLocalSALADHead.__init__,
        "mlx_forward": MLXLocalSALADHead.__call__, "mlx_dropout": _PrivateMLXDropout.__call__,
    }.items()}
    return {
        "loaded_code": loaded_codes,
        "schema": TRAINING_IDENTITY_VERSION,
        "files": {name: sha256_file(root / name) for name in
                  ("local_salad.py", "local_salad_mlx.py", "local_salad_provenance.py")},
        "packages": {name: _version(name) for name in ("torch", "mlx", "numpy")},
        "seed_normalization": "signed_integer_modulo_2pow32_private_initialization_and_mlx_key",
        "seed_policy": "private_head_initialization_and_dropout_rng_v2",
        "determinism": "seeded; numerical reproducibility must be measured per runtime",
    }


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


@contextmanager
def qualified_training_rng(seed, backend="torch"):
    """Serialize head trainers without reading/changing process-global RNG.

    Trainers pass seed explicitly to the existing head constructor. Private
    initialization and dropout streams live on that head; augmentations use
    existing independent Random instances. Unrelated framework draws neither
    perturb training nor get rolled back. This lock owns SALAD trainer capacity,
    not the process-global RNG of unrelated workflows.
    """
    with TRAINING_RNG_LOCK:
        yield {"seed": int(seed), "backend": backend,
               **training_implementation_identity()}
