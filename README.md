
# High-Order Mutual Information Codes

This folder contains Python scripts to compute the $n$th-order mutual information for the frustrated Ising ring:

- **MI.py**: full thermal ensemble computation for $I_N$ vs $(\beta J)$ with multiple $N$.
- **MI_gs.py**: ground-state manifold computation ($\beta J \to -\infty$), with optional acceleration and analytic curve overlay.

## Requirements

- Python 3.8+
- `numpy`
- `matplotlib`
- `numba` (optional, for faster ground-state combinatorics in MI_gs.py)

Install dependencies in your environment:

```
pip install numpy matplotlib numba
```

## Usage

### 1) Thermal ensemble (MI.py)

Edit hard-coded parameters in `main()`:

- `ns`: list of system sizes to overlay
- `j0`: coupling $J$
- `betaj_min`, `betaj_max`: range for $(\beta J)$
- `num`: number of points
- `save_path`: output file name (e.g., `In.pdf`)

Run:

```
python MI.py
```

The plot is saved to `save_path`.

### 2) Ground-state manifold (MI_gs.py)

Edit hard-coded parameters in `main()`:

- `n_min`, `n_max`, `n_step`: range of $N$
- `log_base`: log base (default 2)
- `save_path`: output file name
- `use_numba`: set `True` if `numba` is installed

Run:

```
python MI_gs.py
```

This produces a plot of $I_N$ vs $N$ (markers only), overlays the analytic formula up to $N=100$, and includes the dashed line at $-\log(e/2)$.

## Notes

- For the frustrated Ising ring, the ground-state manifold consists of $2N$ configurations with exactly one frustrated bond.
- The analytic formula overlay in `MI_gs.py` uses:

$$
I_N = -\log\left[\frac{1}{2}\left(1 + \frac{1}{N-1}\right)^{N-1}\right]
$$

with the log base set by `log_base`.

