"""Unit tests for the OFI computation.

The toy book below is small enough to compute OFI by hand, which pins down the
Cont-Kukanov-Stoikov sign conventions and catches any regression in the math.
"""
import numpy as np
import pandas as pd

from src.ofi import compute_ofi


def test_ofi_known_values():
    # A hand-worked sequence of best-quote updates.
    df = pd.DataFrame({
        "bid_px": [99.99, 99.99, 100.00, 100.00, 100.00],
        "bid_sz": [10,    15,    8,      8,      8],
        "ask_px": [100.01, 100.01, 100.01, 100.02, 100.01],
        "ask_sz": [10,     10,     10,     12,     5],
    })
    # n=1: bid size +5, ask unchanged                  -> +5
    # n=2: bid price up (dW=+8), ask unchanged         -> +8
    # n=3: ask price up (dV=-10), bid unchanged        -> +10
    # n=4: ask price down (dV=+5), bid unchanged       -> -5
    expected = np.array([0.0, 5.0, 8.0, 10.0, -5.0])
    assert np.allclose(compute_ofi(df), expected)


def test_ofi_first_element_zero():
    df = pd.DataFrame({"bid_px": [1.0, 1.0], "bid_sz": [5, 7],
                       "ask_px": [2.0, 2.0], "ask_sz": [5, 5]})
    assert compute_ofi(df)[0] == 0.0


def test_ofi_length_matches_input():
    df = pd.DataFrame({"bid_px": [1.0, 1.0, 1.0], "bid_sz": [5, 6, 7],
                       "ask_px": [2.0, 2.0, 2.0], "ask_sz": [5, 5, 5]})
    assert len(compute_ofi(df)) == len(df)
