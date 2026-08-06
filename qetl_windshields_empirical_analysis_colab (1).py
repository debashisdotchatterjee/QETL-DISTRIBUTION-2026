
# ============================================================
# QETL EMPIRICAL ANALYSIS: AIRCRAFT WINDSHIELD SERVICE TIMES
# ============================================================
# Focused, Colab-ready application for the 63 service times of
# aircraft windshields selected by the preceding named-data screen.
#
# Main empirical message:
#   * Gompertz is the AICc-leading model.
#   * QETL is second with Delta AICc < 2.
#   * QETL decisively improves on the Lindley boundary.
#
# Outputs:
#   * all tables printed inline and saved as CSV/LaTeX;
#   * all figures printed inline and saved as PNG/PDF;
#   * boundary LRT and parametric bootstrap;
#   * QETL bootstrap intervals and GOF p-values;
#   * automatic downloadable ZIP.
# ============================================================

import math
import json
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize, brentq
from scipy.special import erfcx, gammaln
from scipy.stats import (
    expon, weibull_min, gamma as gamma_dist,
    lognorm, gompertz, gengamma, chi2
)
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------- Settings ----------------
SEED = 20260728
FAST_MODE = True
BOOTSTRAP_REP = 400 if FAST_MODE else 1000
DPI = 240
RNG = np.random.default_rng(SEED)

ROOT = Path("QETL_WINDSHIELDS_APPLICATION")
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
DATA_DIR = ROOT / "data"
for d in (ROOT, FIG_DIR, TAB_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8.4, 5.2),
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

# ---------------- Exact selected data ----------------
# Service times of 63 aircraft windshields.
x = np.array([
    0.046, 1.436, 2.592,
    0.140, 1.492, 2.600,
    0.150, 1.580, 2.670,
    0.248, 1.719, 2.717,
    0.280, 1.794, 2.819,
    0.313, 1.915, 2.820,
    0.389, 1.920, 2.878,
    0.487, 1.963, 2.950,
    0.622, 1.978, 3.003,
    0.900, 2.053, 3.102,
    0.952, 2.065, 3.304,
    0.996, 2.117, 3.483,
    1.003, 2.137, 3.500,
    1.010, 2.141, 3.622,
    1.085, 2.163, 3.665,
    1.092, 2.183, 3.695,
    1.152, 2.240, 4.015,
    1.183, 2.341, 4.628,
    1.244, 2.435, 4.806,
    1.249, 2.464, 4.881,
    1.262, 2.543, 5.140
], dtype=float)

assert len(x) == 63 and np.all(x > 0)
pd.DataFrame({"service_time": x}).to_csv(
    DATA_DIR / "aircraft_windshield_service_times.csv", index=False
)

# ============================================================
# QETL distribution
# ============================================================

def J_sequence(theta, alpha, max_r=5):
    if theta <= 0 or alpha < 0:
        raise ValueError("theta > 0 and alpha >= 0 required")
    J = np.zeros(max_r + 1)
    if alpha <= 1e-12:
        for r in range(max_r + 1):
            J[r] = math.exp(gammaln(r + 1) - (r + 1) * math.log(theta))
        return J
    z = theta / (2 * math.sqrt(alpha))
    J[0] = math.sqrt(math.pi) * erfcx(z) / (2 * math.sqrt(alpha))
    J[1] = (1 - theta * J[0]) / (2 * alpha)
    for r in range(1, max_r):
        J[r + 1] = (r * J[r - 1] - theta * J[r]) / (2 * alpha)
    return J

def qetl_Z(theta, alpha):
    J = J_sequence(theta, alpha, 1)
    return J[0] + J[1]

def qetl_logpdf(z, theta, alpha):
    z = np.asarray(z, float)
    if theta <= 0 or alpha < 0 or np.any(z <= 0):
        return np.full_like(z, -np.inf)
    return np.log1p(z) - theta*z - alpha*z*z - math.log(qetl_Z(theta, alpha))

def qetl_pdf(z, theta, alpha):
    return np.exp(qetl_logpdf(z, theta, alpha))

def qetl_survival(z, theta, alpha):
    z = np.asarray(z, float)
    out = np.ones_like(z)
    mask = z > 0
    zz = z[mask]
    if alpha <= 1e-12:
        out[mask] = (theta + 1 + theta*zz) * np.exp(-theta*zz) / (theta + 1)
    else:
        a = math.sqrt(alpha)
        A = (
            math.sqrt(math.pi)/(2*a)
            * np.exp(-alpha*zz*zz - theta*zz)
            * erfcx(a*zz + theta/(2*a))
        )
        B = (np.exp(-theta*zz - alpha*zz*zz) - theta*A) / (2*alpha)
        out[mask] = (A + B) / qetl_Z(theta, alpha)
    return np.clip(out, 0, 1)

def qetl_cdf(z, theta, alpha):
    return 1 - qetl_survival(z, theta, alpha)

def qetl_hazard(z, theta, alpha):
    z = np.asarray(z, float)
    S = qetl_survival(z, theta, alpha)
    return np.divide(
        qetl_pdf(z, theta, alpha), S,
        out=np.full_like(z, np.nan), where=S > 1e-14
    )

def qetl_ppf(p, theta, alpha):
    p = np.asarray(p, float)
    ans = np.empty_like(p)
    for ind, pp in np.ndenumerate(p):
        if pp <= 0:
            ans[ind] = 0
            continue
        if pp >= 1:
            ans[ind] = np.inf
            continue
        hi = max(2.0, 2.0/theta)
        while qetl_cdf(np.array([hi]), theta, alpha)[0] < pp:
            hi *= 2
        ans[ind] = brentq(
            lambda q: qetl_cdf(np.array([q]), theta, alpha)[0] - pp,
            0, hi
        )
    return ans

def qetl_moments(theta, alpha):
    J = J_sequence(theta, alpha, 5)
    Z = J[0] + J[1]
    raw = [(J[r] + J[r+1])/Z for r in range(1, 5)]
    mu = raw[0]
    var = raw[1] - mu**2
    skew = (raw[2] - 3*mu*raw[1] + 2*mu**3) / var**1.5
    kurt = (
        raw[3] - 4*mu*raw[2] + 6*mu**2*raw[1] - 3*mu**4
    ) / var**2
    return mu, var, skew, kurt

def lindley_mle(z):
    n = len(z)
    sz = np.sum(z)
    return (-(sz-n) + math.sqrt((sz-n)**2 + 8*n*sz)) / (2*sz)

# ============================================================
# Boundary-aware QETL maximum likelihood
# ============================================================

def fit_qetl(z):
    z = np.asarray(z, float)
    theta_L = lindley_mle(z)
    ll0 = float(np.sum(qetl_logpdf(z, theta_L, 0.0)))

    candidates = [{
        "theta": theta_L, "alpha": 0.0,
        "logLik": ll0, "location": "boundary"
    }]

    starts = [
        (theta_L, 0.005),
        (theta_L, 0.03),
        (theta_L, 0.10),
        (theta_L, 0.50),
        (max(0.02, theta_L*0.5), 1.0),
        (theta_L*1.5, 0.05)
    ]

    for theta0, alpha0 in starts:
        def objective(eta):
            theta, alpha = np.exp(eta)
            ll = np.sum(qetl_logpdf(z, theta, alpha))
            return 1e100 if not np.isfinite(ll) else -ll

        res = minimize(
            objective, np.log([theta0, alpha0]),
            method="Nelder-Mead",
            options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-10}
        )
        theta, alpha = np.exp(res.x)
        ll = -res.fun
        if np.isfinite(ll):
            candidates.append({
                "theta": theta, "alpha": alpha,
                "logLik": ll, "location": "interior"
            })

    best = max(candidates, key=lambda d: d["logLik"])
    best["lindley_logLik"] = ll0
    best["loglik_gain"] = best["logLik"] - ll0
    return best

# ============================================================
# Fit QETL and benchmark models
# ============================================================

def edf_statistics(z, cdf_fun):
    zs = np.sort(z)
    n = len(zs)
    F = np.clip(cdf_fun(zs), 1e-12, 1-1e-12)
    i = np.arange(1, n+1)
    KS = max(np.max(i/n - F), np.max(F - (i-1)/n))
    CvM = 1/(12*n) + np.sum((F - (2*i-1)/(2*n))**2)
    AD = -n - np.mean((2*i-1)*(np.log(F) + np.log(1-F[::-1])))
    return KS, CvM, AD

def information_criteria(logLik, k, n):
    AIC = -2*logLik + 2*k
    AICc = AIC + 2*k*(k+1)/(n-k-1)
    BIC = -2*logLik + k*np.log(n)
    HQIC = -2*logLik + 2*k*np.log(np.log(n))
    return AIC, AICc, BIC, HQIC

scale = np.mean(x)
y = x / scale
n = len(y)
fits = []

def add_fit(name, k, ll_y, params, cdf_y, ppf_y, pdf_y, hazard_y):
    ll = ll_y - n*np.log(scale)
    KS, CvM, AD = edf_statistics(y, cdf_y)
    AIC, AICc, BIC, HQIC = information_criteria(ll, k, n)
    fits.append({
        "model": name, "k": k, "logLik": ll,
        "AIC": AIC, "AICc": AICc, "BIC": BIC, "HQIC": HQIC,
        "KS": KS, "CvM": CvM, "AD": AD,
        "parameters": params,
        "cdf_y": cdf_y, "ppf_y": ppf_y,
        "pdf_y": pdf_y, "hazard_y": hazard_y
    })

# Exponential
exp_sc = np.mean(y)
add_fit(
    "Exponential", 1,
    np.sum(expon.logpdf(y, loc=0, scale=exp_sc)),
    {"scale_original": exp_sc*scale},
    lambda z: expon.cdf(z, loc=0, scale=exp_sc),
    lambda p: expon.ppf(p, loc=0, scale=exp_sc),
    lambda z: expon.pdf(z, loc=0, scale=exp_sc),
    lambda z: np.full_like(np.asarray(z, float), 1/exp_sc)
)

# Lindley
theta_L = lindley_mle(y)
ll_L = np.sum(qetl_logpdf(y, theta_L, 0))
add_fit(
    "Lindley", 1, ll_L,
    {"theta_scaled": theta_L, "theta_original": theta_L/scale},
    lambda z: qetl_cdf(z, theta_L, 0),
    lambda p: qetl_ppf(p, theta_L, 0),
    lambda z: qetl_pdf(z, theta_L, 0),
    lambda z: qetl_hazard(z, theta_L, 0)
)

# QETL
fq = fit_qetl(y)
theta_Q, alpha_Q = fq["theta"], fq["alpha"]
add_fit(
    "QETL", 2, fq["logLik"],
    {
        "theta_scaled": theta_Q,
        "alpha_scaled": alpha_Q,
        "theta_original": theta_Q/scale,
        "alpha_original": alpha_Q/scale**2,
        "fit_location": fq["location"],
        "loglik_gain_over_Lindley": fq["loglik_gain"]
    },
    lambda z: qetl_cdf(z, theta_Q, alpha_Q),
    lambda p: qetl_ppf(p, theta_Q, alpha_Q),
    lambda z: qetl_pdf(z, theta_Q, alpha_Q),
    lambda z: qetl_hazard(z, theta_Q, alpha_Q)
)

# Standard competitors
for name, dist, k in [
    ("Weibull", weibull_min, 2),
    ("Gamma", gamma_dist, 2),
    ("Lognormal", lognorm, 2),
    ("Gompertz", gompertz, 2),
    ("Generalized gamma", gengamma, 3)
]:
    pars = dist.fit(y, floc=0)
    ll = np.sum(dist.logpdf(y, *pars))
    add_fit(
        name, k, ll,
        {"scipy_parameters": tuple(float(v) for v in pars)},
        lambda z, d=dist, p=pars: d.cdf(z, *p),
        lambda u, d=dist, p=pars: d.ppf(u, *p),
        lambda z, d=dist, p=pars: d.pdf(z, *p),
        lambda z, d=dist, p=pars: np.divide(
            d.pdf(z, *p), d.sf(z, *p),
            out=np.full_like(np.asarray(z, float), np.nan),
            where=d.sf(z, *p) > 1e-14
        )
    )

fit_map = {f["model"]: f for f in fits}

# ============================================================
# Table utility
# ============================================================

def display_save_table(df, stem, title, label):
    display(Markdown(f"## {title}"))
    display(df)
    df.to_csv(TAB_DIR/f"{stem}.csv", index=False)
    df.to_latex(
        TAB_DIR/f"{stem}.tex",
        index=False,
        float_format=lambda z: f"{z:.6f}",
        caption=title,
        label=label,
        escape=False
    )

# Descriptive statistics
summary = pd.DataFrame([{
    "n": len(x),
    "minimum": np.min(x),
    "Q1": np.quantile(x, 0.25),
    "median": np.median(x),
    "mean": np.mean(x),
    "Q3": np.quantile(x, 0.75),
    "maximum": np.max(x),
    "SD": np.std(x, ddof=1),
    "CV": np.std(x, ddof=1)/np.mean(x),
    "skewness": pd.Series(x).skew(),
    "kurtosis": pd.Series(x).kurt()+3
}])
display_save_table(
    summary, "table_01_data_summary",
    "Aircraft-windshield service-time summary",
    "tab:windshield-summary"
)

# Model comparison
comparison = pd.DataFrame([
    {k:v for k,v in f.items()
     if k not in {"parameters","cdf_y","ppf_y","pdf_y","hazard_y"}}
    for f in fits
]).sort_values("AICc").reset_index(drop=True)

comparison["Delta_AICc"] = comparison["AICc"] - comparison["AICc"].min()
comparison["Akaike_weight"] = np.exp(-0.5*comparison["Delta_AICc"])
comparison["Akaike_weight"] /= comparison["Akaike_weight"].sum()
comparison["Evidence_ratio"] = comparison["Akaike_weight"].iloc[0] / comparison["Akaike_weight"]
comparison["AICc_rank"] = np.arange(1, len(comparison)+1)

display_save_table(
    comparison, "table_02_model_comparison",
    "Maximum-likelihood comparison of fitted lifetime models",
    "tab:windshield-model-comparison"
)

# Parameters
param_rows = []
for f in fits:
    row = {"model": f["model"]}
    row.update({k:str(v) for k,v in f["parameters"].items()})
    param_rows.append(row)
parameter_table = pd.DataFrame(param_rows)
display_save_table(
    parameter_table, "table_03_parameter_estimates",
    "Maximum-likelihood parameter estimates",
    "tab:windshield-parameters"
)

# Empirical vs QETL moments
mu_q, var_q, skew_q, kurt_q = qetl_moments(theta_Q, alpha_Q)
moment_table = pd.DataFrame([
    {"quantity":"Mean", "empirical":np.mean(x), "QETL_fitted":mu_q*scale},
    {"quantity":"Variance", "empirical":np.var(x, ddof=1), "QETL_fitted":var_q*scale**2},
    {"quantity":"Skewness", "empirical":pd.Series(x).skew(), "QETL_fitted":skew_q},
    {"quantity":"Kurtosis", "empirical":pd.Series(x).kurt()+3, "QETL_fitted":kurt_q}
])
display_save_table(
    moment_table, "table_04_empirical_fitted_moments",
    "Empirical and QETL-fitted moments",
    "tab:windshield-moments"
)

# ============================================================
# Boundary test and bootstrap
# ============================================================

LR_obs = 2*(fq["logLik"] - ll_L)
mixture_p = 0.5*chi2.sf(LR_obs, 1) if LR_obs > 0 else 1.0

def r_lindley(size, theta, rng):
    mix = rng.random(size) < theta/(theta+1)
    z = np.empty(size)
    z[mix] = rng.exponential(1/theta, mix.sum())
    z[~mix] = rng.gamma(shape=2, scale=1/theta, size=(~mix).sum())
    return z

def r_qetl(size, theta, alpha, rng):
    if alpha <= 1e-12:
        return r_lindley(size, theta, rng)
    pacc = qetl_Z(theta, alpha)/((theta+1)/theta**2)
    pieces = []
    total = 0
    while total < size:
        batch = max(500, int((size-total)/max(pacc, 0.03)*1.08))
        z = r_lindley(batch, theta, rng)
        keep = rng.random(batch) < np.exp(-alpha*z*z)
        pieces.append(z[keep])
        total += keep.sum()
    return np.concatenate(pieces)[:size]

# Parametric bootstrap under Lindley boundary
LR_null = np.empty(BOOTSTRAP_REP)
for b in range(BOOTSTRAP_REP):
    yb = r_lindley(n, theta_L, RNG)
    th0 = lindley_mle(yb)
    ll0 = np.sum(qetl_logpdf(yb, th0, 0))
    qfb = fit_qetl(yb)
    LR_null[b] = max(0, 2*(qfb["logLik"] - ll0))

bootstrap_LR_p = (1 + np.sum(LR_null >= LR_obs))/(BOOTSTRAP_REP + 1)

test_table = pd.DataFrame([{
    "LR_statistic": LR_obs,
    "mixture_chi_square_p": mixture_p,
    "bootstrap_p": bootstrap_LR_p,
    "bootstrap_replications": BOOTSTRAP_REP
}])
display_save_table(
    test_table, "table_05_lindley_boundary_test",
    "Likelihood-ratio test of Lindley versus QETL",
    "tab:windshield-lr-test"
)

# QETL bootstrap inference and GOF
qrow = comparison[comparison["model"]=="QETL"].iloc[0]
boot = np.empty((BOOTSTRAP_REP, 6))

for b in range(BOOTSTRAP_REP):
    yb = r_qetl(n, theta_Q, alpha_Q, RNG)
    qfb = fit_qetl(yb)
    KS, CvM, AD = edf_statistics(
        yb, lambda z: qetl_cdf(z, qfb["theta"], qfb["alpha"])
    )
    boot[b] = [
        qfb["theta"], qfb["alpha"], KS, CvM, AD,
        1 if qfb["location"]=="boundary" else 0
    ]

boot_interval = pd.DataFrame([
    {
        "parameter":"theta_scaled",
        "estimate":theta_Q,
        "bootstrap_SE":np.std(boot[:,0], ddof=1),
        "CI_2.5":np.quantile(boot[:,0], .025),
        "CI_97.5":np.quantile(boot[:,0], .975)
    },
    {
        "parameter":"alpha_scaled",
        "estimate":alpha_Q,
        "bootstrap_SE":np.std(boot[:,1], ddof=1),
        "CI_2.5":np.quantile(boot[:,1], .025),
        "CI_97.5":np.quantile(boot[:,1], .975)
    }
])
display_save_table(
    boot_interval, "table_06_qetl_bootstrap_intervals",
    "QETL parametric-bootstrap uncertainty",
    "tab:windshield-bootstrap"
)

gof_boot = pd.DataFrame([{
    "KS_observed":qrow["KS"],
    "KS_bootstrap_p":np.mean(boot[:,2] >= qrow["KS"]),
    "CvM_observed":qrow["CvM"],
    "CvM_bootstrap_p":np.mean(boot[:,3] >= qrow["CvM"]),
    "AD_observed":qrow["AD"],
    "AD_bootstrap_p":np.mean(boot[:,4] >= qrow["AD"]),
    "boundary_fit_frequency":np.mean(boot[:,5]),
    "replications":BOOTSTRAP_REP
}])
display_save_table(
    gof_boot, "table_07_qetl_bootstrap_gof",
    "QETL bootstrap goodness-of-fit diagnostics",
    "tab:windshield-gof"
)

# ============================================================
# Figures
# ============================================================

def save_show(fig, stem):
    fig.savefig(FIG_DIR/f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR/f"{stem}.pdf", bbox_inches="tight")
    plt.show()
    plt.close(fig)

models = list(comparison["model"][:6])
xs = np.sort(x)
emp = np.arange(1, n+1)/n
p_emp = (np.arange(1, n+1)-0.5)/n
grid = np.linspace(1e-6, x.max()*1.08, 1000)
gscaled = grid/scale

# 1. Histogram and densities
fig, ax = plt.subplots(figsize=(8.7,5.3))
ax.hist(x, bins="fd", density=True, alpha=.40, label="Observed data")
for m in models:
    dens = fit_map[m]["pdf_y"](gscaled)/scale
    ax.plot(grid, dens, lw=1.8, label=m)
ax.set(title="Aircraft-windshield service times: fitted densities",
       xlabel="Service time", ylabel="Density")
ax.grid(alpha=.2)
ax.legend(frameon=False, ncol=2)
save_show(fig, "figure_01_fitted_densities")

# 2. CDF and survival
fig, axes = plt.subplots(1,2,figsize=(10.8,4.4))
axes[0].step(xs, emp, where="post", lw=2, label="Empirical")
axes[1].step(xs, 1-emp, where="post", lw=2, label="Empirical")
for m in models:
    F = fit_map[m]["cdf_y"](gscaled)
    axes[0].plot(grid, F, label=m)
    axes[1].plot(grid, 1-F, label=m)
axes[0].set(title="Distribution functions", xlabel="Service time", ylabel="$F(x)$")
axes[1].set(title="Survival functions", xlabel="Service time", ylabel="$S(x)$")
for ax in axes:
    ax.grid(alpha=.2)
axes[0].legend(frameon=False, fontsize=7, ncol=2)
save_show(fig, "figure_02_cdf_survival")

# 3. Hazard comparison
fig, ax = plt.subplots(figsize=(8.3,5.1))
for m in models[:5]:
    h = fit_map[m]["hazard_y"](gscaled)/scale
    ax.plot(grid, h, lw=1.8, label=m)
ax.set(title="Fitted hazard functions",
       xlabel="Service time", ylabel="Hazard")
ax.grid(alpha=.2)
ax.legend(frameon=False)
save_show(fig, "figure_03_hazard_comparison")

# 4. P-P and Q-Q
fig, axes = plt.subplots(1,2,figsize=(10.7,4.5))
axes[0].plot([0,1],[0,1],"--",lw=1.2)
axes[1].plot([xs.min(),xs.max()],[xs.min(),xs.max()],"--",lw=1.2)
for m in models:
    axes[0].plot(
        p_emp, fit_map[m]["cdf_y"](xs/scale),
        marker="o", ms=2.4, lw=1, label=m
    )
    axes[1].plot(
        fit_map[m]["ppf_y"](p_emp)*scale, xs,
        marker="o", ms=2.4, lw=1, label=m
    )
axes[0].set(title="P--P plot", xlabel="Empirical probability",
            ylabel="Fitted probability")
axes[1].set(title="Q--Q plot", xlabel="Fitted quantiles",
            ylabel="Observed quantiles")
for ax in axes:
    ax.grid(alpha=.2)
axes[0].legend(frameon=False, fontsize=7)
save_show(fig, "figure_04_pp_qq")

# 5. TTT plot
spacings = np.diff(np.r_[0, xs])
ttt = np.cumsum((n-np.arange(n))*spacings)
phi = ttt/ttt[-1]
u = np.arange(1,n+1)/n

fig, ax = plt.subplots(figsize=(6.4,5.1))
ax.plot(u, phi, lw=2, label="Empirical scaled TTT")
ax.plot([0,1],[0,1],"--",label="Exponential reference")
ax.set(title="Scaled total-time-on-test transform",
       xlabel="$i/n$", ylabel="$T(i/n)$")
ax.grid(alpha=.2)
ax.legend(frameon=False)
save_show(fig, "figure_05_ttt")

# 6. Cox-Snell residuals for top four
fig, axes = plt.subplots(2,2,figsize=(9.4,7.1))
for ax, m in zip(axes.flat, models[:4]):
    F = np.clip(fit_map[m]["cdf_y"](y), 1e-12, 1-1e-12)
    r = np.sort(-np.log(1-F))
    H = -np.log(1-(np.arange(1,n+1)-0.5)/n)
    ax.scatter(r,H,s=18)
    lim = max(r.max(),H.max())
    ax.plot([0,lim],[0,lim],"--")
    ax.set(title=m, xlabel="Cox--Snell residual",
           ylabel="Empirical cumulative hazard")
    ax.grid(alpha=.2)
fig.suptitle("Cox--Snell residual diagnostics")
save_show(fig, "figure_06_cox_snell")

# 7. AICc comparison
fig, ax = plt.subplots(figsize=(7.8,5.0))
plot_df = comparison.sort_values("Delta_AICc", ascending=True)
ax.barh(plot_df["model"], plot_df["Delta_AICc"])
ax.invert_yaxis()
ax.axvline(2, linestyle="--", linewidth=1.2)
ax.set(title="Relative AICc support",
       xlabel=r"$\Delta$AICc", ylabel="Model")
ax.grid(axis="x", alpha=.2)
save_show(fig, "figure_07_delta_aicc")

# 8. Likelihood-ratio bootstrap
fig, ax = plt.subplots(figsize=(7.4,5.0))
ax.hist(LR_null, bins=30, density=True, alpha=.45,
        label="Bootstrap null distribution")
ax.axvline(LR_obs, linestyle="--", linewidth=2,
           label=f"Observed LR = {LR_obs:.3f}")
ax.set(title="Boundary likelihood-ratio calibration",
       xlabel="Likelihood-ratio statistic", ylabel="Density")
ax.grid(alpha=.2)
ax.legend(frameon=False)
save_show(fig, "figure_08_lr_bootstrap")

# 9. QETL bootstrap parameter cloud
fig, ax = plt.subplots(figsize=(6.5,5.2))
ax.scatter(boot[:,0], boot[:,1], s=14, alpha=.4)
ax.scatter([theta_Q],[alpha_Q], marker="*", s=150,
           label="Observed-data MLE")
ax.set(title="QETL bootstrap parameter estimates",
       xlabel=r"$\widehat\theta$ (scaled)",
       ylabel=r"$\widehat\alpha$ (scaled)")
ax.grid(alpha=.2)
ax.legend(frameon=False)
save_show(fig, "figure_09_bootstrap_cloud")

# ============================================================
# Final summary and ZIP
# ============================================================

q_comp = comparison[comparison["model"]=="QETL"].iloc[0]
g_comp = comparison[comparison["model"]=="Gompertz"].iloc[0]
l_comp = comparison[comparison["model"]=="Lindley"].iloc[0]

key_findings = pd.DataFrame([{
    "selected_dataset": "Aircraft windshield service times",
    "n": n,
    "best_model_by_AICc": comparison.iloc[0]["model"],
    "QETL_AICc_rank": int(q_comp["AICc_rank"]),
    "QETL_Delta_AICc": q_comp["Delta_AICc"],
    "QETL_Akaike_weight": q_comp["Akaike_weight"],
    "QETL_vs_Lindley_loglik_gain": fq["loglik_gain"],
    "QETL_vs_Lindley_LR": LR_obs,
    "mixture_p": mixture_p,
    "bootstrap_p": bootstrap_LR_p,
    "QETL_alpha_scaled": alpha_Q
}])
display_save_table(
    key_findings, "table_08_key_findings",
    "Key empirical findings",
    "tab:windshield-key-findings"
)

metadata = {
    "dataset": "Service times of 63 aircraft windshields",
    "seed": SEED,
    "fast_mode": FAST_MODE,
    "bootstrap_replications": BOOTSTRAP_REP,
    "best_model_AICc": comparison.iloc[0]["model"],
    "QETL_rank_AICc": int(q_comp["AICc_rank"]),
    "QETL_delta_AICc": float(q_comp["Delta_AICc"]),
    "QETL_akaike_weight": float(q_comp["Akaike_weight"]),
    "QETL_vs_Lindley_loglik_gain": float(fq["loglik_gain"]),
    "LR_statistic": float(LR_obs),
    "mixture_p": float(mixture_p),
    "bootstrap_p": float(bootstrap_LR_p)
}
(ROOT/"metadata.json").write_text(json.dumps(metadata, indent=2))

out_zip = Path("QETL_WINDSHIELDS_EMPIRICAL_RESULTS.zip")
if out_zip.exists():
    out_zip.unlink()

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in ROOT.rglob("*"):
        if path.is_file():
            zf.write(path, path.as_posix())

print("\n" + "="*72)
print("EMPIRICAL ANALYSIS COMPLETE")
print("="*72)
print("Dataset: service times of 63 aircraft windshields")
print(f"Best AICc model: {comparison.iloc[0]['model']}")
print(f"QETL AICc rank: {int(q_comp['AICc_rank'])}")
print(f"QETL Delta AICc: {q_comp['Delta_AICc']:.6f}")
print(f"QETL Akaike weight: {q_comp['Akaike_weight']:.6f}")
print(f"QETL vs Lindley log-likelihood gain: {fq['loglik_gain']:.6f}")
print(f"Boundary LR statistic: {LR_obs:.6f}")
print(f"Mixture-law p-value: {mixture_p:.6g}")
print(f"Bootstrap p-value: {bootstrap_LR_p:.6g}")
print(f"Created: {out_zip.resolve()}")

try:
    from google.colab import files
    files.download(str(out_zip))
except Exception:
    print("Not running in Colab; download the ZIP from the working directory.")
