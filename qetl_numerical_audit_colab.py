# -*- coding: utf-8 -*-
"""
Numerical audit for the Quadratically Exponentially Tilted Lindley (QETL) law.

Designed for Google Colab and standard Python 3.10+ environments.
The script:
  1. verifies density normalization;
  2. verifies -S'(x)=f(x);
  3. verifies M^(r)(0)=E(X^r), r=1,...,4;
  4. verifies the J_r moment recursion;
  5. verifies positive definiteness of Fisher information;
  6. compares exact rejection samples with the theoretical CDF;
  7. prints every table and displays every figure inline;
  8. saves all tables/figures and creates a downloadable ZIP archive.

No external data or internet connection is required.
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import integrate, special, stats

try:
    import mpmath as mp
except ImportError as exc:
    raise ImportError("mpmath is required. In Colab run: !pip install mpmath") from exc

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)

# -----------------------------------------------------------------------------
# Reproducibility and output configuration
# -----------------------------------------------------------------------------
SEED = 20260727
RNG = np.random.default_rng(SEED)
OUTPUT_DIR = Path("QETL_NUMERICAL_AUDIT")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# Moderate settings covering low/high theta and weak/strong quadratic tilting.
PARAMETER_GRID: tuple[tuple[float, float], ...] = (
    (0.60, 0.08),
    (0.90, 0.35),
    (1.50, 0.15),
    (2.50, 0.80),
    (4.00, 1.50),
)

# Sampling settings are intentionally moderate for fast Colab execution.
SAMPLE_SETTINGS: tuple[tuple[float, float], ...] = (
    (0.60, 0.08),
    (1.50, 0.15),
    (2.50, 0.80),
    (4.00, 1.50),
)
N_SAMPLE = 20_000
QUAD_EPSABS = 2e-12
QUAD_EPSREL = 2e-12


@dataclass(frozen=True)
class QETL:
    theta: float
    alpha: float

    def __post_init__(self) -> None:
        if not (self.theta > 0.0):
            raise ValueError("theta must be strictly positive")
        if not (self.alpha >= 0.0):
            raise ValueError("alpha must be nonnegative")

    @property
    def z0(self) -> float:
        """Lindley boundary normalizer Z(theta, 0)."""
        return (self.theta + 1.0) / self.theta**2

    def A(self, x: np.ndarray | float) -> np.ndarray | float:
        """Stable A_x = integral_x^infinity exp(-alpha*u^2-theta*u) du."""
        x_arr = np.asarray(x, dtype=float)
        if self.alpha == 0.0:
            out = np.exp(-self.theta * x_arr) / self.theta
        else:
            sa = math.sqrt(self.alpha)
            z = sa * x_arr + self.theta / (2.0 * sa)
            # Algebraically identical to exp(theta^2/(4a))*erfc(z), but stable.
            out = (
                math.sqrt(math.pi)
                / (2.0 * sa)
                * np.exp(-self.alpha * x_arr**2 - self.theta * x_arr)
                * special.erfcx(z)
            )
        return float(out) if np.ndim(x) == 0 else out

    def Z_closed(self) -> float:
        """Closed-form normalizer, with quadrature fallback near cancellation."""
        if self.alpha == 0.0:
            return self.z0
        a0 = float(self.A(0.0))
        j1 = (1.0 - self.theta * a0) / (2.0 * self.alpha)
        z = a0 + j1
        # For unusually ill-conditioned settings, use direct positive quadrature.
        if (not np.isfinite(z)) or z <= 0.0 or abs(1.0 - self.theta * a0) < 5e-11:
            z = self.Z_quad()
        return float(z)

    def Z_quad(self) -> float:
        val, _ = integrate.quad(
            lambda u: (1.0 + u) * math.exp(-self.theta * u - self.alpha * u * u),
            0.0,
            np.inf,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=250,
        )
        return float(val)

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        x_arr = np.asarray(x, dtype=float)
        out = np.where(
            x_arr >= 0.0,
            (1.0 + x_arr)
            * np.exp(-self.theta * x_arr - self.alpha * x_arr**2)
            / self.Z_closed(),
            0.0,
        )
        return float(out) if np.ndim(x) == 0 else out

    def survival(self, x: np.ndarray | float) -> np.ndarray | float:
        """Stable closed survival formula; exact Lindley boundary included."""
        x_arr = np.asarray(x, dtype=float)
        if self.alpha == 0.0:
            out = (
                (self.theta + 1.0 + self.theta * x_arr)
                / (self.theta + 1.0)
                * np.exp(-self.theta * x_arr)
            )
        else:
            q = self.alpha * x_arr**2 + self.theta * x_arr
            numerator = np.exp(-q) + (2.0 * self.alpha - self.theta) * self.A(x_arr)
            out = numerator / (2.0 * self.alpha * self.Z_closed())
        out = np.where(x_arr < 0.0, 1.0, np.clip(out, 0.0, 1.0))
        return float(out) if np.ndim(x) == 0 else out

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        out = 1.0 - np.asarray(self.survival(x))
        out = np.clip(out, 0.0, 1.0)
        return float(out) if np.ndim(x) == 0 else out

    def J_quad(self, r: int) -> float:
        val, _ = integrate.quad(
            lambda u: u**r * math.exp(-self.theta * u - self.alpha * u * u),
            0.0,
            np.inf,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=250,
        )
        return float(val)

    def J_recursion(self, max_r: int) -> np.ndarray:
        """Compute J_0,...,J_max_r from the exact recurrence."""
        if max_r < 1:
            raise ValueError("max_r must be at least 1")
        if self.alpha == 0.0:
            return np.array([math.gamma(r + 1.0) / self.theta ** (r + 1) for r in range(max_r + 1)])
        js = np.empty(max_r + 1, dtype=float)
        js[0] = float(self.A(0.0))
        js[1] = (1.0 - self.theta * js[0]) / (2.0 * self.alpha)
        for r in range(1, max_r):
            js[r + 1] = (r * js[r - 1] - self.theta * js[r]) / (2.0 * self.alpha)
        return js

    def raw_moment_quad(self, r: int) -> float:
        val, _ = integrate.quad(
            lambda u: u**r * self.pdf(u),
            0.0,
            np.inf,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=250,
        )
        return float(val)

    def raw_moments(self, max_r: int = 4) -> np.ndarray:
        return np.array([self.raw_moment_quad(r) for r in range(max_r + 1)])

    def fisher_information(self) -> np.ndarray:
        mu = self.raw_moments(4)
        return np.array(
            [
                [mu[2] - mu[1] ** 2, mu[3] - mu[1] * mu[2]],
                [mu[3] - mu[1] * mu[2], mu[4] - mu[2] ** 2],
            ],
            dtype=float,
        )

    def lindley_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Exact Lindley mixture: Exp(theta) or Gamma(shape=2, rate=theta)."""
        component = rng.random(n) < self.theta / (self.theta + 1.0)
        out = np.empty(n, dtype=float)
        n_exp = int(component.sum())
        out[component] = rng.exponential(scale=1.0 / self.theta, size=n_exp)
        out[~component] = rng.gamma(shape=2.0, scale=1.0 / self.theta, size=n - n_exp)
        return out

    def rejection_sample(
        self, n: int, rng: np.random.Generator, batch_size: int | None = None
    ) -> tuple[np.ndarray, float, int]:
        """Exact rejection sampler with vectorized Lindley proposals."""
        if self.alpha == 0.0:
            return self.lindley_sample(n, rng), 1.0, n
        theoretical_rate = self.Z_closed() / self.z0
        if batch_size is None:
            batch_size = max(2_000, int(math.ceil(1.20 * n / theoretical_rate)))
        accepted_parts: list[np.ndarray] = []
        accepted_total = 0
        proposed_total = 0
        while accepted_total < n:
            y = self.lindley_sample(batch_size, rng)
            accept = rng.random(batch_size) <= np.exp(-self.alpha * y**2)
            vals = y[accept]
            accepted_parts.append(vals)
            accepted_total += vals.size
            proposed_total += batch_size
            remaining = n - accepted_total
            if remaining > 0:
                batch_size = max(2_000, int(math.ceil(1.15 * remaining / theoretical_rate)))
        sample = np.concatenate(accepted_parts)[:n]
        empirical_rate = accepted_total / proposed_total
        return sample, float(empirical_rate), proposed_total


# -----------------------------------------------------------------------------
# High-precision independent MGF differentiation
# -----------------------------------------------------------------------------
def mp_Z(theta: float | mp.mpf, alpha: float | mp.mpf) -> mp.mpf:
    theta_mp = mp.mpf(theta)
    alpha_mp = mp.mpf(alpha)
    if alpha_mp == 0:
        return (theta_mp + 1) / theta_mp**2
    a0 = (
        mp.sqrt(mp.pi)
        / (2 * mp.sqrt(alpha_mp))
        * mp.exp(theta_mp**2 / (4 * alpha_mp))
        * mp.erfc(theta_mp / (2 * mp.sqrt(alpha_mp)))
    )
    return a0 + (1 - theta_mp * a0) / (2 * alpha_mp)


def mgf_derivative_at_zero(theta: float, alpha: float, r: int) -> float:
    mp.mp.dps = 60
    z = mp_Z(theta, alpha)
    fun = lambda t: mp_Z(mp.mpf(theta) - t, mp.mpf(alpha)) / z
    return float(mp.diff(fun, mp.mpf("0.0"), r))


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def save_show(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)


def scientific_table(df: pd.DataFrame, digits: int = 6) -> pd.io.formats.style.Styler:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    fmt = {col: (lambda x, d=digits: f"{x:.{d}e}") for col in numeric_cols}
    return df.style.format(fmt).hide(axis="index")


# -----------------------------------------------------------------------------
# Audit calculations
# -----------------------------------------------------------------------------
def audit_normalization() -> pd.DataFrame:
    rows = []
    for theta, alpha in PARAMETER_GRID:
        model = QETL(theta, alpha)
        integral, err = integrate.quad(
            model.pdf,
            0.0,
            np.inf,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=250,
        )
        rows.append(
            {
                "theta": theta,
                "alpha": alpha,
                "integral_f": integral,
                "abs_error": abs(integral - 1.0),
                "quad_error_bound": err,
                "Z_closed": model.Z_closed(),
                "Z_quadrature": model.Z_quad(),
                "relative_Z_error": abs(model.Z_closed() / model.Z_quad() - 1.0),
            }
        )
    return pd.DataFrame(rows)


def audit_survival_derivative() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []
    for theta, alpha in PARAMETER_GRID:
        model = QETL(theta, alpha)
        # Grid adapts to the distribution using high quantiles from a pilot sample.
        pilot, _, _ = model.rejection_sample(4_000, RNG)
        xmax = float(np.quantile(pilot, 0.995))
        x_grid = np.linspace(max(1e-4, xmax / 500), xmax, 180)
        h = max(2e-5, 3e-5 * (1.0 + xmax))
        # Five-point central derivative, kept away from x=0.
        x_grid = x_grid[x_grid > 2.2 * h]
        deriv = (
            -model.survival(x_grid + 2 * h)
            + 8 * model.survival(x_grid + h)
            - 8 * model.survival(x_grid - h)
            + model.survival(x_grid - 2 * h)
        ) / (12 * h)
        lhs = -deriv
        rhs = model.pdf(x_grid)
        abs_err = np.abs(lhs - rhs)
        rel_err = abs_err / np.maximum(rhs, 1e-11)
        summary_rows.append(
            {
                "theta": theta,
                "alpha": alpha,
                "max_abs_error": float(abs_err.max()),
                "median_abs_error": float(np.median(abs_err)),
                "max_relative_error_pdf_gt_1e-8": float(rel_err[rhs > 1e-8].max()),
            }
        )
        for x, lv, rv, ae in zip(x_grid, lhs, rhs, abs_err):
            detail_rows.append(
                {
                    "theta": theta,
                    "alpha": alpha,
                    "x": x,
                    "minus_S_prime": lv,
                    "pdf": rv,
                    "abs_error": ae,
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def audit_mgf_derivatives() -> pd.DataFrame:
    rows = []
    for theta, alpha in PARAMETER_GRID:
        model = QETL(theta, alpha)
        for r in range(1, 5):
            moment = model.raw_moment_quad(r)
            derivative = mgf_derivative_at_zero(theta, alpha, r)
            rows.append(
                {
                    "theta": theta,
                    "alpha": alpha,
                    "order_r": r,
                    "quadrature_moment": moment,
                    "mgf_derivative": derivative,
                    "abs_error": abs(moment - derivative),
                    "relative_error": abs(moment - derivative) / max(abs(moment), 1e-15),
                }
            )
    return pd.DataFrame(rows)


def audit_moment_recursion() -> pd.DataFrame:
    rows = []
    max_r = 7
    for theta, alpha in PARAMETER_GRID:
        model = QETL(theta, alpha)
        j_rec = model.J_recursion(max_r)
        for r in range(max_r + 1):
            j_quad = model.J_quad(r)
            rows.append(
                {
                    "theta": theta,
                    "alpha": alpha,
                    "r": r,
                    "J_recursion": j_rec[r],
                    "J_quadrature": j_quad,
                    "abs_error": abs(j_rec[r] - j_quad),
                    "relative_error": abs(j_rec[r] - j_quad) / max(abs(j_quad), 1e-15),
                }
            )
    return pd.DataFrame(rows)


def audit_fisher() -> pd.DataFrame:
    rows = []
    for theta, alpha in PARAMETER_GRID:
        model = QETL(theta, alpha)
        info = model.fisher_information()
        eig = np.linalg.eigvalsh(info)
        rows.append(
            {
                "theta": theta,
                "alpha": alpha,
                "I11": info[0, 0],
                "I12": info[0, 1],
                "I22": info[1, 1],
                "determinant": np.linalg.det(info),
                "min_eigenvalue": eig[0],
                "max_eigenvalue": eig[1],
                "condition_number": eig[1] / eig[0],
                "positive_definite": bool(eig[0] > 0.0),
            }
        )
    return pd.DataFrame(rows)


def audit_sampling() -> tuple[pd.DataFrame, dict[tuple[float, float], np.ndarray]]:
    rows = []
    samples: dict[tuple[float, float], np.ndarray] = {}
    for theta, alpha in SAMPLE_SETTINGS:
        model = QETL(theta, alpha)
        sample, empirical_acc, proposed = model.rejection_sample(N_SAMPLE, RNG)
        samples[(theta, alpha)] = sample
        ordered = np.sort(sample)
        theo = model.cdf(ordered)
        ecdf_upper = np.arange(1, N_SAMPLE + 1) / N_SAMPLE
        ecdf_lower = np.arange(0, N_SAMPLE) / N_SAMPLE
        ks = max(float(np.max(ecdf_upper - theo)), float(np.max(theo - ecdf_lower)))
        cvm = float(np.mean((theo - (np.arange(1, N_SAMPLE + 1) - 0.5) / N_SAMPLE) ** 2) + 1.0 / (12 * N_SAMPLE**2))
        theoretical_acc = model.Z_closed() / model.z0
        rows.append(
            {
                "theta": theta,
                "alpha": alpha,
                "n": N_SAMPLE,
                "proposals": proposed,
                "theoretical_acceptance": theoretical_acc,
                "empirical_acceptance": empirical_acc,
                "acceptance_abs_error": abs(empirical_acc - theoretical_acc),
                "KS_distance": ks,
                "Cramer_von_Mises": cvm,
                "sample_mean": np.mean(sample),
                "theoretical_mean": model.raw_moment_quad(1),
                "sample_variance": np.var(sample, ddof=1),
                "theoretical_variance": model.raw_moment_quad(2) - model.raw_moment_quad(1) ** 2,
            }
        )
    return pd.DataFrame(rows), samples


# -----------------------------------------------------------------------------
# Publication-quality figures
# -----------------------------------------------------------------------------
def plot_survival_identity(detail: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for (theta, alpha), grp in detail.groupby(["theta", "alpha"]):
        ax.plot(grp["x"], grp["abs_error"], label=fr"$\theta={theta:g},\ \alpha={alpha:g}$")
    ax.set_yscale("log")
    ax.set_xlabel("x")
    ax.set_ylabel(r"Absolute error in $-S'(x)=f(x)$")
    ax.set_title("Numerical differentiation audit of the survival identity")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    save_show(fig, "figure_1_survival_derivative_audit")


def plot_mgf_and_recursion(mgf_df: pd.DataFrame, rec_df: pd.DataFrame) -> None:
    combined = pd.DataFrame(
        {
            "check": ["MGF derivatives", "Moment recursion"],
            "max_relative_error": [mgf_df["relative_error"].max(), rec_df["relative_error"].max()],
        }
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(combined["check"], combined["max_relative_error"])
    ax.set_yscale("log")
    ax.set_ylabel("Maximum relative error")
    ax.set_title("Independent transform and moment-recursion audits")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for i, val in enumerate(combined["max_relative_error"]):
        ax.text(i, val * 1.25, f"{val:.2e}", ha="center", va="bottom")
    save_show(fig, "figure_2_transform_and_recursion_errors")


def plot_fisher_eigenvalues(fisher_df: pd.DataFrame) -> None:
    labels = [fr"$({t:g},{a:g})$" for t, a in zip(fisher_df.theta, fisher_df.alpha)]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(x, fisher_df["min_eigenvalue"], marker="o", label="Minimum eigenvalue")
    ax.plot(x, fisher_df["max_eigenvalue"], marker="s", label="Maximum eigenvalue")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_xlabel(r"Parameter pair $(\theta,\alpha)$")
    ax.set_ylabel("Eigenvalue")
    ax.set_title("Positive definiteness of the Fisher information")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    save_show(fig, "figure_3_fisher_information_eigenvalues")


def plot_cdf_agreement(samples: dict[tuple[float, float], np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for (theta, alpha), sample in samples.items():
        model = QETL(theta, alpha)
        x = np.linspace(0.0, np.quantile(sample, 0.995), 240)
        ordered = np.sort(sample)
        idx = np.searchsorted(ordered, x, side="right")
        ecdf = idx / ordered.size
        ax.plot(x, model.cdf(x), linewidth=2.0, label=fr"Theory: $({theta:g},{alpha:g})$")
        ax.plot(x, ecdf, linestyle="--", linewidth=1.2, label=fr"Sample: $({theta:g},{alpha:g})$")
    ax.set_xlabel("x")
    ax.set_ylabel("Distribution function")
    ax.set_title("Exact rejection samples versus theoretical QETL CDFs")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save_show(fig, "figure_4_empirical_and_theoretical_cdfs")


def plot_probability_integral_transform(samples: dict[tuple[float, float], np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    probs = np.linspace(0.01, 0.99, 99)
    for (theta, alpha), sample in samples.items():
        u = np.sort(QETL(theta, alpha).cdf(sample))
        empirical_quantiles = np.quantile(u, probs)
        ax.plot(probs, empirical_quantiles, label=fr"$({theta:g},{alpha:g})$")
    ax.plot([0, 1], [0, 1], linestyle=":", linewidth=2.0, label="Uniform reference")
    ax.set_xlabel("Uniform probability")
    ax.set_ylabel("Empirical PIT quantile")
    ax.set_title("Probability-integral-transform diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_show(fig, "figure_5_probability_integral_transform")


# -----------------------------------------------------------------------------
# Output management
# -----------------------------------------------------------------------------
def save_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    df.to_latex(TABLE_DIR / f"{stem}.tex", index=False, float_format="%.8e", escape=False)


def write_summary(
    runtime: float,
    normalization: pd.DataFrame,
    survival: pd.DataFrame,
    mgf: pd.DataFrame,
    recursion: pd.DataFrame,
    fisher: pd.DataFrame,
    sampling: pd.DataFrame,
) -> None:
    summary = {
        "seed": SEED,
        "runtime_seconds": runtime,
        "python": sys.version,
        "platform": platform.platform(),
        "max_normalization_error": float(normalization["abs_error"].max()),
        "max_survival_derivative_abs_error": float(survival["max_abs_error"].max()),
        "max_mgf_derivative_relative_error": float(mgf["relative_error"].max()),
        "max_moment_recursion_relative_error": float(recursion["relative_error"].max()),
        "minimum_fisher_eigenvalue": float(fisher["min_eigenvalue"].min()),
        "all_fisher_matrices_positive_definite": bool(fisher["positive_definite"].all()),
        "maximum_sampling_KS_distance": float(sampling["KS_distance"].max()),
    }
    (OUTPUT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "QETL NUMERICAL AUDIT SUMMARY",
        "=" * 34,
        f"Runtime: {runtime:.2f} seconds",
        f"Maximum |integral f - 1|: {summary['max_normalization_error']:.3e}",
        f"Maximum absolute error in -S'=f: {summary['max_survival_derivative_abs_error']:.3e}",
        f"Maximum relative error in MGF derivative identity: {summary['max_mgf_derivative_relative_error']:.3e}",
        f"Maximum relative error in J recursion: {summary['max_moment_recursion_relative_error']:.3e}",
        f"Minimum Fisher-information eigenvalue: {summary['minimum_fisher_eigenvalue']:.3e}",
        f"All Fisher matrices positive definite: {summary['all_fisher_matrices_positive_definite']}",
        f"Maximum rejection-sample KS distance: {summary['maximum_sampling_KS_distance']:.3e}",
    ]
    (OUTPUT_DIR / "README_RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def create_zip() -> Path:
    # Include the running source file when available.
    source = Path(__file__).resolve() if "__file__" in globals() else None
    if source and source.exists():
        shutil.copy2(source, OUTPUT_DIR / source.name)
    zip_path = Path("QETL_NUMERICAL_AUDIT_RESULTS.zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(zip_path.with_suffix("").as_posix(), "zip", root_dir=OUTPUT_DIR)
    return zip_path


def print_table(title: str, df: pd.DataFrame, digits: int = 6) -> None:
    print("\n" + title)
    print("-" * len(title))
    display(scientific_table(df, digits=digits))


def main() -> None:
    start = time.perf_counter()
    print("QETL numerical audit")
    print(f"Random seed: {SEED}")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    normalization = audit_normalization()
    print_table("Table 1. Density normalization and normalizing constant", normalization)
    save_table(normalization, "table_1_normalization")

    survival_summary, survival_detail = audit_survival_derivative()
    print_table("Table 2. Verification of -S'(x) = f(x)", survival_summary)
    save_table(survival_summary, "table_2_survival_derivative_summary")
    survival_detail.to_csv(TABLE_DIR / "table_2_survival_derivative_full_grid.csv", index=False)

    mgf = audit_mgf_derivatives()
    print_table("Table 3. Verification of M^(r)(0) = E(X^r)", mgf)
    save_table(mgf, "table_3_mgf_derivatives")

    recursion = audit_moment_recursion()
    print_table("Table 4. Verification of the J_r moment recursion", recursion)
    save_table(recursion, "table_4_moment_recursion")

    fisher = audit_fisher()
    print_table("Table 5. Fisher-information positive-definiteness audit", fisher)
    save_table(fisher, "table_5_fisher_information")

    sampling, samples = audit_sampling()
    print_table("Table 6. Rejection-sampling and CDF agreement", sampling)
    save_table(sampling, "table_6_rejection_sampling")

    plot_survival_identity(survival_detail)
    plot_mgf_and_recursion(mgf, recursion)
    plot_fisher_eigenvalues(fisher)
    plot_cdf_agreement(samples)
    plot_probability_integral_transform(samples)

    runtime = time.perf_counter() - start
    write_summary(runtime, normalization, survival_summary, mgf, recursion, fisher, sampling)
    zip_path = create_zip()
    print(f"\nZIP archive created: {zip_path.resolve()}")

    # Trigger browser download in Google Colab; harmlessly skipped elsewhere.
    try:
        from google.colab import files  # type: ignore

        files.download(str(zip_path))
    except Exception:
        print("Not running in Google Colab: download the ZIP from the working directory.")


if __name__ == "__main__":
    main()
