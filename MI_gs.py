import itertools
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def generate_ground_states(n: int) -> np.ndarray:
    """Generate frustrated Ising ring ground states with exactly one domain wall."""
    states: List[Tuple[int, ...]] = []

    if n % 2 == 1:
        # Odd N: one frustrated bond in an AFM ring (2N states)
        for parity in (0, 1):
            for k in range(1, n + 1):
                spins: List[int] = []
                for i in range(1, n + 1):
                    if i <= k:
                        spins.append((-1) ** (i + parity))
                    else:
                        spins.append((-1) ** (i + parity + 1))
                states.append(tuple(spins))
    else:
        # Even N: twisted PBC ground states
        for p in (0, 1):
            spins = [(-1) ** (i - 1 + p) for i in range(1, n + 1)]
            states.append(tuple(spins))

        for p in (0, 1):
            for k in range(1, n):
                spins = [(-1) ** (i - 1 + p) for i in range(1, k + 1)]
                spins.append(spins[k - 1])
                for _i in range(k + 2, n + 1):
                    spins.append(-spins[-1])
                states.append(tuple(spins))

    # Remove duplicates while preserving order
    unique_states = list(dict.fromkeys(states))
    return np.array(unique_states, dtype=int)


def _encode_subset(bits: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    subset = list(subset)
    powers = (1 << np.arange(len(subset), dtype=np.uint32))
    return bits[:, subset] @ powers


def compute_in_ground_states(
    ground_states: np.ndarray,
    log_base: float = 2.0,
    show_progress: bool = True,
) -> float:
    m, n = ground_states.shape
    bits = (ground_states < 0).astype(np.uint8)

    subsets_by_size: Dict[int, List[Tuple[int, ...]]] = {
        size: list(itertools.combinations(range(n), size)) for size in range(1, n + 1)
    }

    if show_progress:
        print(f"Total ground states: {m}")
        print("Precomputing subset marginals...")

    subset_logp: Dict[Tuple[int, ...], np.ndarray] = {}
    for size in range(1, n + 1):
        for subset in subsets_by_size[size]:
            codes = _encode_subset(bits, subset)
            probs = np.bincount(codes, minlength=1 << size) / m
            subset_logp[subset] = np.log(probs[codes])

    log_q = np.zeros(m)
    for k in range(2, n + 1):
        if show_progress:
            print(f"Computing Q step k={k}/{n}")
        log_prod = np.zeros(m)
        for subset in subsets_by_size[k - 1]:
            log_prod += subset_logp[subset]
        log_q = log_prod - log_q

    log_p = -math.log(m)
    inv_log_base = 1.0 / math.log(log_base)
    return -np.sum((1.0 / m) * (log_p - log_q)) * inv_log_base


if _NUMBA_AVAILABLE:

    @njit
    def _compute_in_ground_states_numba(bits_vals: np.ndarray, n: int, log_base: float) -> float:
        m = bits_vals.shape[0]
        log_q = np.zeros(m, dtype=np.float64)

        for k in range(2, n + 1):
            log_prod = np.zeros(m, dtype=np.float64)
            comb = np.arange(k - 1, dtype=np.int64)

            while True:
                counts = np.zeros(1 << (k - 1), dtype=np.int64)
                codes = np.empty(m, dtype=np.uint32)

                for i in range(m):
                    b = bits_vals[i]
                    code = 0
                    for t in range(k - 1):
                        pos = comb[t]
                        code |= ((b >> pos) & 1) << t
                    codes[i] = code
                    counts[code] += 1

                for i in range(m):
                    log_prod[i] += math.log(counts[codes[i]] / m)

                i = k - 2
                while i >= 0 and comb[i] == i + n - (k - 1):
                    i -= 1
                if i < 0:
                    break
                comb[i] += 1
                for j in range(i + 1, k - 1):
                    comb[j] = comb[j - 1] + 1

            log_q = log_prod - log_q

        log_p = -math.log(m)
        inv_log_base = 1.0 / math.log(log_base)
        return -np.sum((1.0 / m) * (log_p - log_q)) * inv_log_base


def compute_in_ground_states_fast(
    ground_states: np.ndarray,
    log_base: float = 2.0,
    show_progress: bool = True,
) -> float:
    if not _NUMBA_AVAILABLE:
        if show_progress:
            print("Numba not available; falling back to Python implementation.")
        return compute_in_ground_states(ground_states, log_base=log_base, show_progress=show_progress)

    bits_vals = (ground_states < 0).astype(np.uint32)
    bits_vals = (bits_vals * (1 << np.arange(bits_vals.shape[1], dtype=np.uint32))).sum(axis=1)
    return _compute_in_ground_states_numba(bits_vals, ground_states.shape[1], log_base)


def main() -> None:
    # Hard-coded parameters (edit here)
    n_min = 3
    n_max = 25
    n_step = 1
    log_base = 2.0
    show_progress = True
    save_path = "IN_vs_N_gs.pdf"
    use_numba = True

    ns = list(range(n_min, n_max + 1, n_step))
    values: List[float] = []

    total_n = len(ns)
    for idx, n in enumerate(ns, start=1):
        if show_progress:
            print(f"Computing N={n} ({idx}/{total_n})")
        ground_states = generate_ground_states(n)
        if use_numba:
            in_value = compute_in_ground_states_fast(
                ground_states, log_base=log_base, show_progress=False
            )
        else:
            in_value = compute_in_ground_states(
                ground_states, log_base=log_base, show_progress=False
            )
        values.append(in_value)

    plt.figure(figsize=(8, 6))
    y_ref = -math.log(math.e / 2.0) / math.log(log_base)
    plt.axhline(y_ref, color="k", lw=1.2, ls="--", alpha=0.8, label=r"$-\log(e/2)$")
    plt.plot(ns, -np.abs(values), marker="o", linestyle="None")
    formula_ns = np.arange(3, 101, 1)
    formula_vals = -np.log(0.5 * (1.0 + 1.0 / (formula_ns - 1)) ** (formula_ns - 1)) / math.log(
        log_base
    )
    plt.plot(formula_ns, formula_vals, lw=1.5, ls="-.", label=r"$-\log\left[\frac{1}{2}\left(1+\frac{1}{N-1}\right)^{N-1}\right]$")
    plt.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _pos: rf"${x:g}$"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _pos: rf"${y:g}$"))
    plt.xlabel(r"$N$")
    plt.ylabel(rf"$I_N$ (log base {log_base:g})")
    plt.ylim(ymax=-0.1)
    plt.xscale('log')
    plt.title(r"Ground-state manifold: $I_N$ vs $N$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)


if __name__ == "__main__":
    main()
