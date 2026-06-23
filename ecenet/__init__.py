"""ECENet — an SO(3)-equivariant interatomic potential (MLIP) using per-edge
SO(2) features.

Public API:
    ECENet            — the model (message passing on when n_mp >= 2)
    ECENetCalculator  — ASE calculator wrapper (lazy import; needs `ase`)
"""

from ecenet.model import ECENet

__all__ = ["ECENet", "ECENetCalculator"]


def __getattr__(name):
    # Lazy so `import ecenet` doesn't require ASE unless the calculator is used.
    if name == "ECENetCalculator":
        from ecenet.calculator import ECENetCalculator
        return ECENetCalculator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
