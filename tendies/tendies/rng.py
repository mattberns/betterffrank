"""mulberry32, implemented identically in Python and JavaScript.

The point is not randomness quality, it is REPRODUCIBILITY ACROSS LANGUAGES.
With the same seed, `web/rng.js` yields the same stream as this module, so a
simulation run in the browser and the same simulation run in Python produce the
same sampled draft, pick for pick. That turns the parity test from "these two
distributions look similar" into an exact assertion, which is the only kind
worth having on draft day.

It also means the page is stable: the same board state always shows the same
survival numbers, because the seed is derived from the state. Numbers that
jitter when nothing changed look broken.

All arithmetic is masked to 32 bits to mirror JS's `| 0` and `Math.imul`.
"""

from __future__ import annotations

M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    """Math.imul: 32-bit integer multiply with wraparound."""
    return ((a & M32) * (b & M32)) & M32


class Mulberry32:
    def __init__(self, seed: int) -> None:
        self.s = seed & M32

    def next_u32(self) -> int:
        self.s = (self.s + 0x6D2B79F5) & M32
        t = self.s
        t = _imul(t ^ (t >> 15), 1 | t)
        # canonical mulberry32 XORs t back in here (`t ^= t + imul(...)`).
        # Dropping that XOR still produces a fine-looking random stream, which
        # is exactly why it needs a parity test rather than an eyeball.
        t = (t ^ ((t + _imul(t ^ (t >> 7), 61 | t)) & M32)) & M32
        t = t ^ (t >> 14)
        return t & M32

    def random(self) -> float:
        return self.next_u32() / 4294967296.0

    def pick(self, probs) -> int:
        """Sample an index from a probability vector (assumed to sum to ~1)."""
        u = self.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if u < acc:
                return i
        return len(probs) - 1


def hash32(text: str) -> int:
    """FNV-1a. Used to fold a state description into a seed."""
    h = 0x811C9DC5
    for ch in text:
        h ^= ord(ch) & 0xFF
        h = _imul(h, 0x01000193)
    return h & M32
