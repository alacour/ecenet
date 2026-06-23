"""Tests for the shared element ↔ type-index mechanism (ecenet/elements.py)."""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ecenet import elements as el


def test_symbol_number_roundtrip():
    for z in (1, 6, 8, 14, 26, 53):
        assert el.number(el.symbol(z)) == z
    assert el.symbol(6) == 'C'
    assert el.number('Cl') == 17
    print("  symbol/number roundtrip OK")


def test_build_type_map_dense_and_sorted():
    # Unsorted, with duplicates and a gap (no element 7).
    nums = [8, 1, 6, 6, 1, 8, 1]
    tm = el.build_type_map(nums)
    # Dense 0..n-1 over the 3 distinct elements, ordered by atomic number.
    assert tm == {1: 0, 6: 1, 8: 2}
    assert sorted(tm.values()) == [0, 1, 2]
    print(f"  build_type_map: {nums} -> {tm}")


def test_build_type_map_is_deterministic():
    a = el.build_type_map([14, 6, 1, 8])
    b = el.build_type_map([8, 1, 14, 6])   # different order, same set
    assert a == b, "ordering must not depend on input order (DDP determinism)"
    print("  build_type_map deterministic across input order")


def test_representation_conversions():
    tm = {1: 0, 6: 1, 8: 2}
    e2t = el.to_element_to_type(tm)
    assert e2t == {'H': 0, 'C': 1, 'O': 2}
    # Round-trips back to the atomic-number form.
    assert el.to_type_map(e2t) == tm
    print(f"  conversions: {tm} <-> {e2t}")


if __name__ == '__main__':
    print("ecenet.elements")
    test_symbol_number_roundtrip()
    test_build_type_map_dense_and_sorted()
    test_build_type_map_is_deterministic()
    test_representation_conversions()
    print("All tests passed.")
