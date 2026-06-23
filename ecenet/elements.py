"""Single source of truth for element ↔ type-index handling.

Every trainer, script, dataset loader, and the calculator goes through here, so
there is exactly one periodic-table lookup and one notion of how a set of
elements becomes type indices.

Type maps are kept **compact**: a dense ``0..n_types-1`` indexing over only the
elements actually present, *not* a fixed global table. This is deliberate —
model size scales with ``n_types`` (the type-conditioned contraction weight is
``n_types²``), so a 4-element molecule must not pay for 100+ absent elements.

Two equivalent representations appear:
  * ``type_map``         — ``{atomic_number: idx}`` (how the data, which carries
    atomic numbers, is tensorised during training)
  * ``element_to_type``  — ``{symbol: idx}`` (the canonical checkpoint form; the
    calculator works in chemical symbols)
"""

from ase.data import atomic_numbers as _Z_BY_SYMBOL  # {symbol: Z}
from ase.data import chemical_symbols as _SYMBOL_BY_Z  # [Z] -> symbol


def symbol(z):
    """Chemical symbol for an atomic number (e.g. ``6 -> 'C'``)."""
    return _SYMBOL_BY_Z[int(z)]


def number(sym):
    """Atomic number for a chemical symbol (e.g. ``'C' -> 6``)."""
    return _Z_BY_SYMBOL[sym]


def build_type_map(atomic_numbers):
    """Dense ``{atomic_number: idx}`` over the distinct elements present.

    Sorted by atomic number, so the result is deterministic — every DDP rank
    and re-run agrees on the ordering. ``atomic_numbers`` is any iterable of
    atomic numbers (duplicates are fine).
    """
    zs = sorted({int(z) for z in atomic_numbers})
    return {z: i for i, z in enumerate(zs)}


def to_element_to_type(type_map):
    """``{atomic_number: idx}`` → ``{symbol: idx}`` (the checkpoint form)."""
    return {symbol(z): idx for z, idx in type_map.items()}


def to_type_map(element_to_type):
    """``{symbol: idx}`` → ``{atomic_number: idx}``."""
    return {number(sym): idx for sym, idx in element_to_type.items()}
