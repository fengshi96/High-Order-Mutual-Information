import itertools
import math
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _configs_and_spins(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return configs as uint ints and spins as +/-1 array (shape: 2^n, n)."""
    configs = np.arange(1 << n, dtype=np.uint32)
    # bit 0 -> spin +1, bit 1 -> spin -1
    bits = (configs[:, None] >> np.arange(n)) & 1
    spins = 1 - 2 * bits
    return configs, spins


def _energy(spins: np.ndarray, j: float) -> np.ndarray:
    n = spins.shape[1]
    neighbor_sum = (spins[:, :-1] * spins[:, 1:]).sum(axis=1)
    boundary = ((-1) ** (n - 1)) * spins[:, 0] * spins[:, -1]
    return -j * (neighbor_sum + boundary)


def _stable_probs(energies: np.ndarray, beta: float) -> Tuple[np.ndarray, float]:
    logw = -beta * energies
    maxw = logw.max()
    w = np.exp(logw - maxw)
    z = w.sum()
    logz = maxw + math.log(z)
    p = np.exp(logw - logz)
    return p, logz


def _subset_index(configs: np.ndarray, subset: Sequence[int]) -> np.ndarray:
    idx = np.zeros_like(configs, dtype=np.uint32)
    for pos, bit in enumerate(subset):
        idx |= ((configs >> bit) & 1) << pos
    return idx


def mutual_information_n(n: int, beta: float, j: float, log_base: float = 2.0) -> float:
    """
    Compute n-th order mutual information I_n for the frustrated Ising chain.

    Parameters
    ----------
    n : int
        Number of spins (odd n recommended per model description).
    beta : float
        Inverse temperature.
    j : float
        Coupling J. For anti-ferromagnetic, J < 0.
    log_base : float
        Logarithm base for I_n (default 2).
    """
    configs, spins = _configs_and_spins(n)
    energies = _energy(spins, j)
    p, _logz = _stable_probs(energies, beta)

    log_q = np.zeros_like(p)
    for k in range(2, n + 1):
        log_prod = np.zeros_like(p)
        for subset in itertools.combinations(range(n), k - 1):
            idx = _subset_index(configs, subset)
            m = np.bincount(idx, weights=p, minlength=1 << (k - 1))
            # m[idx] is the marginal probability for the subset config
            log_prod += np.log(m[idx])
        log_q = log_prod - log_q

    # I_n = -sum P * log(P / Q) = -sum P * (logP - logQ)
    log_p = np.log(p)
    inv_log_base = 1.0 / math.log(log_base)
    return -np.sum(p * (log_p - log_q)) * inv_log_base


def _prepare_cache(n: int, j: float) -> dict:
    configs, spins = _configs_and_spins(n)
    energies = _energy(spins, j)
    subset_indices_by_k: List[Tuple[List[np.ndarray], int]] = []
    for k in range(2, n + 1):
        indices: List[np.ndarray] = []
        for subset in itertools.combinations(range(n), k - 1):
            indices.append(_subset_index(configs, subset))
        subset_indices_by_k.append((indices, 1 << (k - 1)))
    return {
        "n": n,
        "energies": energies,
        "subset_indices_by_k": subset_indices_by_k,
    }


def mutual_information_n_cached(cache: dict, beta: float, log_base: float = 2.0) -> float:
    energies = cache["energies"]
    p, _logz = _stable_probs(energies, beta)

    log_q = np.zeros_like(p)
    for indices, m_len in cache["subset_indices_by_k"]:
        log_prod = np.zeros_like(p)
        for idx in indices:
            m = np.bincount(idx, weights=p, minlength=m_len)
            log_prod += np.log(m[idx])
        log_q = log_prod - log_q

    log_p = np.log(p)
    inv_log_base = 1.0 / math.log(log_base)
    return -np.sum(p * (log_p - log_q)) * inv_log_base


def mutual_information_curve_cached(
    cache: dict, betas: np.ndarray, log_base: float = 2.0
) -> np.ndarray:
    energies = cache["energies"]
    values = np.empty_like(betas, dtype=float)
    inv_log_base = 1.0 / math.log(log_base)

    for i, beta in enumerate(betas):
        p, _logz = _stable_probs(energies, beta)

        log_q = np.zeros_like(p)
        for indices, m_len in cache["subset_indices_by_k"]:
            log_prod = np.zeros_like(p)
            for idx in indices:
                m = np.bincount(idx, weights=p, minlength=m_len)
                log_prod += np.log(m[idx])
            log_q = log_prod - log_q

        log_p = np.log(p)
        values[i] = -np.sum(p * (log_p - log_q)) * inv_log_base

    return values


def plot_in_vs_beta(
    ns: Sequence[int],
    j0: float = -1.0,
    betaj_min: float = -2.0,
    betaj_max: float = 2.0,
    num: int = 200,
    log_base: float = 2.0,
    save_path: str | None = None,
) -> None:
    """Plot I_N as a function of beta*J for multiple N on the same axes."""
    if j0 == 0:
        raise ValueError("j0 must be nonzero to plot versus beta*J.")
    betaj = np.linspace(betaj_min, betaj_max, num=num)
    betas = betaj / j0

    plt.figure(figsize=(8, 6))
    total_n = len(ns)
    for idx, n in enumerate(ns, start=1):
        print(f"Computing N={n} ({idx}/{total_n})")
        cache = _prepare_cache(n, j0)
        values = mutual_information_curve_cached(cache, betas, log_base=log_base)
        print(f"I_{n} at min beta J={betaj_min:g}: {values[0]}")
        plt.plot(betaj, values, lw=2, label=rf"$N={n}$")

    plt.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    plt.xlabel(r"$\beta J$")
    plt.ylabel(rf"$I_N$ (log base {log_base:g})")
    plt.title(rf"Frustrated Ising chain: $I_N(\beta J)$, $J={j0:g}$")
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()


def main() -> None:
    # Hard-coded parameters (edit here)
    ns = [3, 5, 7, 9, 11, 13]
    j0 = -1.0
    betaj_min = -4.0
    betaj_max = 4.0
    num = 100
    log_base = 2.0
    save_path = "In_thermal.pdf"  

    plot_in_vs_beta(
        ns=ns,
        j0=j0,
        betaj_min=betaj_min,
        betaj_max=betaj_max,
        num=num,
        log_base=log_base,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
