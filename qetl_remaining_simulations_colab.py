
# ============================================================
# QETL DISTRIBUTION: DISTRIBUTIONAL ATLAS + INFERENCE SIMULATION
# Colab-ready, self-contained, fast, and publication-oriented
#
# Model:
#   f(x; theta, alpha) =
#       (1+x) exp(-theta*x - alpha*x^2) / Z(theta, alpha),
#   x > 0, theta > 0, alpha >= 0.
#
# This program deliberately does NOT repeat the earlier numerical
# formula audit. It studies:
#   1. Density, CDF, survival, hazard, cumulative hazard
#   2. Quantiles and tail concentration
#   3. Mean residual life and mean inactivity time
#   4. Moment-based shape measures, entropy and inequality
#   5. Parameter-effect surfaces and limiting behaviour
#   6. Exact random-sample diagnostics
#   7. Finite-sample MLE bias, RMSE, coverage and convergence
#   8. Likelihood contours and estimator correlation
#   9. Boundary behaviour when alpha = 0
#
# All tables and plots are:
#   - printed/displayed inline;
#   - saved as CSV and LaTeX;
#   - saved as PNG and PDF;
#   - collected in a ZIP archive;
#   - automatically downloaded in Google Colab.
# ============================================================

import os
import time
import math
import json
import shutil
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize, brentq
from scipy.special import erfcx, gammaln
from scipy.stats import norm, chi2
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------
# User-adjustable settings
# ---------------------------
SEED = 20260728
FAST_MODE = True      # True: journal-quality diagnostic run in a few minutes
N_REP = 160 if FAST_MODE else 500
N_SAMPLE_SHAPE = 30000 if FAST_MODE else 100000
GRID_SIZE = 900
DPI = 220

RNG = np.random.default_rng(SEED)

ROOT = Path("QETL_REMAINING_SIMULATIONS")
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
DATA_DIR = ROOT / "data"
for d in (ROOT, FIG_DIR, TAB_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (8.2, 5.2),
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

# ============================================================
# Core distribution functions
# ============================================================

def _check_params(theta, alpha):
    if theta <= 0 or alpha < 0:
        raise ValueError("Require theta > 0 and alpha >= 0.")

def lindley_Z(theta):
    return (theta + 1.0) / theta**2

def J_sequence(theta, alpha, max_r=8):
    """
    J_r(theta, alpha) = integral_0^infty x^r exp(-theta*x-alpha*x^2) dx.
    For alpha=0 use Gamma integral. For alpha>0 use J0/J1 plus recursion.
    """
    _check_params(theta, alpha)
    J = np.zeros(max_r + 1, dtype=float)
    if alpha == 0:
        for r in range(max_r + 1):
            J[r] = math.exp(gammaln(r + 1) - (r + 1) * math.log(theta))
        return J

    z = theta / (2.0 * math.sqrt(alpha))
    J[0] = math.sqrt(math.pi) * erfcx(z) / (2.0 * math.sqrt(alpha))
    J[1] = (1.0 - theta * J[0]) / (2.0 * alpha)

    for r in range(1, max_r):
        J[r + 1] = (r * J[r - 1] - theta * J[r]) / (2.0 * alpha)

    # Recurrence can lose precision in extreme corners. This study uses
    # moderate parameter ranges; guard only against tiny negative roundoff.
    J[np.abs(J) < 1e-15] = 0.0
    return J

def Z(theta, alpha):
    J = J_sequence(theta, alpha, 1)
    return J[0] + J[1]

def logZ(theta, alpha):
    return math.log(Z(theta, alpha))

def raw_moments(theta, alpha, max_r=6):
    J = J_sequence(theta, alpha, max_r + 1)
    z = J[0] + J[1]
    out = np.ones(max_r + 1)
    for r in range(1, max_r + 1):
        out[r] = (J[r] + J[r + 1]) / z
    return out

def pdf(x, theta, alpha):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = (1.0 + x[mask]) * np.exp(-theta*x[mask] - alpha*x[mask]**2) / Z(theta, alpha)
    return out

def _tail_J0(x, theta, alpha):
    """Integral_x^infty exp(-theta*t-alpha*t^2) dt."""
    x = np.asarray(x, dtype=float)
    if alpha == 0:
        return np.exp(-theta*x) / theta
    a = math.sqrt(alpha)
    z0 = theta / (2*a)
    zx = a*x + z0
    # exp(theta^2/(4a)) erfc(zx) written stably via erfcx
    exponent = -alpha*x*x - theta*x
    return math.sqrt(math.pi)/(2*a) * np.exp(exponent) * erfcx(zx)

def survival(x, theta, alpha):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    mask = x > 0
    xx = x[mask]
    t0 = _tail_J0(xx, theta, alpha)
    if alpha == 0:
        t1 = np.exp(-theta*xx) * (theta*xx + 1.0) / theta**2
    else:
        t1 = (np.exp(-theta*xx-alpha*xx**2) - theta*t0) / (2.0*alpha)
    out[mask] = (t0 + t1) / Z(theta, alpha)
    return np.clip(out, 0.0, 1.0)

def cdf(x, theta, alpha):
    return np.clip(1.0 - survival(x, theta, alpha), 0.0, 1.0)

def hazard(x, theta, alpha):
    s = survival(x, theta, alpha)
    return np.divide(pdf(x, theta, alpha), s, out=np.full_like(s, np.nan), where=s>0)

def cumhaz(x, theta, alpha):
    s = survival(x, theta, alpha)
    return -np.log(np.clip(s, 1e-300, 1.0))

def quantile(p, theta, alpha):
    p = np.asarray(p, dtype=float)
    ans = np.empty_like(p)
    for idx, pp in np.ndenumerate(p):
        if pp <= 0:
            ans[idx] = 0.0
            continue
        if pp >= 1:
            ans[idx] = np.inf
            continue
        hi = max(2.0, 2.0/theta)
        while cdf(np.array([hi]), theta, alpha)[0] < pp:
            hi *= 2.0
        ans[idx] = brentq(lambda q: cdf(np.array([q]), theta, alpha)[0]-pp, 0.0, hi)
    return ans

def shape_measures(theta, alpha):
    m = raw_moments(theta, alpha, 6)
    mu = m[1]
    var = m[2]-mu**2
    sd = math.sqrt(var)
    mu3 = m[3] - 3*mu*m[2] + 2*mu**3
    mu4 = m[4] - 4*mu*m[3] + 6*mu**2*m[2] - 3*mu**4
    skew = mu3/sd**3
    kurt = mu4/var**2
    cv = sd/mu
    q = quantile(np.array([0.1,0.25,0.5,0.75,0.9,0.95,0.99]), theta, alpha)
    return {
        "mean": mu, "variance": var, "sd": sd, "cv": cv,
        "skewness": skew, "kurtosis": kurt, "excess_kurtosis": kurt-3,
        "q10": q[0], "q25": q[1], "median": q[2], "q75": q[3],
        "q90": q[4], "q95": q[5], "q99": q[6],
        "iqr": q[3]-q[1]
    }

def numerical_functionals(theta, alpha, grid_n=5000):
    q9999 = quantile(np.array([0.9999]), theta, alpha)[0]
    x = np.linspace(0.0, q9999, grid_n)
    f = pdf(x, theta, alpha)
    F = cdf(x, theta, alpha)
    S = 1-F

    entropy = -np.trapezoid(np.where(f>0, f*np.log(f), 0.0), x)
    # Gini = 1 - (1/mu) integral S(x)^2 dx for nonnegative X
    mu = raw_moments(theta, alpha, 2)[1]
    gini = 1.0 - np.trapezoid(S**2, x)/mu

    # Mean residual life m(t)= integral_t^infty S(u)du / S(t)
    # Reverse cumulative integral on finite q9999 grid; tail beyond is negligible.
    rev_int_S = cumulative_trapezoid(S[::-1], x[::-1], initial=0.0)[::-1] * (-1)
    mrl = np.divide(rev_int_S, S, out=np.full_like(S, np.nan), where=S>1e-10)

    # Mean inactivity time = integral_0^t F(u)du / F(t)
    int_F = cumulative_trapezoid(F, x, initial=0.0)
    mit = np.divide(int_F, F, out=np.full_like(F, np.nan), where=F>1e-10)

    return entropy, gini, x, mrl, mit

# ============================================================
# Exact rejection sampling
# ============================================================

def r_lindley(n, theta, rng):
    # Lindley = mixture Exp(theta) with prob theta/(theta+1)
    # and Gamma(shape=2, rate=theta) with prob 1/(theta+1).
    mix = rng.random(n) < theta/(theta+1.0)
    y = np.empty(n)
    y[mix] = rng.exponential(scale=1.0/theta, size=mix.sum())
    y[~mix] = rng.gamma(shape=2.0, scale=1.0/theta, size=(~mix).sum())
    return y

def r_qetl(n, theta, alpha, rng=RNG):
    _check_params(theta, alpha)
    if alpha == 0:
        return r_lindley(n, theta, rng)
    accepted = []
    total = 0
    # Expected acceptance = Z(theta,alpha)/Z(theta,0)
    pa = Z(theta, alpha)/lindley_Z(theta)
    batch = max(1000, int((n/max(pa,0.05))*1.08))
    while total < n:
        y = r_lindley(batch, theta, rng)
        keep = rng.random(batch) < np.exp(-alpha*y*y)
        if np.any(keep):
            accepted.append(y[keep])
            total += keep.sum()
        remaining = n-total
        if remaining > 0:
            batch = max(1000, int((remaining/max(pa,0.05))*1.08))
    return np.concatenate(accepted)[:n]

# ============================================================
# Utility: save and display
# ============================================================

def save_table(df, stem, caption=None, label=None, float_format="%.6g"):
    csv_path = TAB_DIR / f"{stem}.csv"
    tex_path = TAB_DIR / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    latex = df.to_latex(
        index=False, escape=False, float_format=lambda z: float_format % z,
        caption=caption, label=label, longtable=(len(df)>25)
    )
    tex_path.write_text(latex, encoding="utf-8")
    display(Markdown(f"### {caption or stem}"))
    display(df)

def save_figure(fig, stem):
    p_png = FIG_DIR / f"{stem}.png"
    p_pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(p_png, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# ============================================================
# 1. Distributional atlas: density, CDF, survival and hazard
# ============================================================

start = time.time()
print("QETL remaining simulation study started.")
print(f"FAST_MODE={FAST_MODE}, N_REP={N_REP}, seed={SEED}")

atlas_settings = [
    (0.6, 0.05), (0.6, 0.40), (0.6, 1.20),
    (1.5, 0.05), (1.5, 0.40), (1.5, 1.20),
    (3.0, 0.05), (3.0, 0.40), (3.0, 1.20),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for theta, alpha in atlas_settings:
    xmax = quantile(np.array([0.995]), theta, alpha)[0]
    x = np.linspace(0, xmax, GRID_SIZE)
    lab = rf"$\theta={theta:g},\ \alpha={alpha:g}$"
    axes[0,0].plot(x, pdf(x,theta,alpha), label=lab)
    axes[0,1].plot(x, cdf(x,theta,alpha), label=lab)
    axes[1,0].plot(x, survival(x,theta,alpha), label=lab)
    axes[1,1].plot(x[1:], hazard(x[1:],theta,alpha), label=lab)

axes[0,0].set(title="Density functions", xlabel="$x$", ylabel="$f(x)$")
axes[0,1].set(title="Distribution functions", xlabel="$x$", ylabel="$F(x)$")
axes[1,0].set(title="Survival functions", xlabel="$x$", ylabel="$S(x)$")
axes[1,1].set(title="Hazard functions", xlabel="$x$", ylabel="$h(x)$")
for ax in axes.flat:
    ax.grid(alpha=0.22)
axes[0,0].legend(ncol=3, frameon=False, fontsize=7.2)
fig.suptitle("QETL distributional atlas across shape and tilt parameters")
save_figure(fig, "figure_01_distributional_atlas")

# Cumulative hazard and log-survival: tail comparison
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
tail_settings = [(0.8,0.0),(0.8,0.15),(0.8,0.6),(1.5,0.0),(1.5,0.6)]
for theta, alpha in tail_settings:
    xmax = quantile(np.array([0.999]), theta, alpha)[0]
    x = np.linspace(0, xmax, GRID_SIZE)
    label = rf"$\theta={theta:g},\alpha={alpha:g}$"
    axes[0].plot(x, cumhaz(x,theta,alpha), label=label)
    axes[1].plot(x, np.log(np.clip(survival(x,theta,alpha),1e-14,1)), label=label)
axes[0].set(title="Cumulative hazard", xlabel="$x$", ylabel="$H(x)$")
axes[1].set(title="Log-survival", xlabel="$x$", ylabel=r"$\log S(x)$")
for ax in axes:
    ax.grid(alpha=0.22)
axes[0].legend(frameon=False, fontsize=8)
save_figure(fig, "figure_02_tail_and_cumulative_hazard")

# ============================================================
# 2. Moments, quantiles, entropy, Gini, and ageing functionals
# ============================================================

summary_settings = [
    (0.5,0.02),(0.5,0.20),(0.5,1.00),
    (1.0,0.02),(1.0,0.20),(1.0,1.00),
    (2.0,0.02),(2.0,0.20),(2.0,1.00),
    (4.0,0.02),(4.0,0.20),(4.0,1.00),
]
rows = []
functional_cache = {}
for theta, alpha in summary_settings:
    d = shape_measures(theta,alpha)
    entropy, gini, gx, mrl, mit = numerical_functionals(theta,alpha,grid_n=3500)
    functional_cache[(theta,alpha)] = (gx,mrl,mit)
    rows.append({
        "theta":theta,"alpha":alpha,
        "mean":d["mean"],"variance":d["variance"],"CV":d["cv"],
        "skewness":d["skewness"],"kurtosis":d["kurtosis"],
        "median":d["median"],"IQR":d["iqr"],"q90":d["q90"],"q99":d["q99"],
        "entropy":entropy,"Gini":gini
    })
shape_df = pd.DataFrame(rows)
save_table(
    shape_df, "table_01_distributional_summaries",
    caption="Moment, quantile, entropy and inequality summaries for representative QETL laws.",
    label="tab:qetl-distributional-summaries"
)

# Heat maps of major shape measures
theta_grid = np.linspace(0.45, 4.0, 24)
alpha_grid = np.linspace(0.02, 1.6, 24)
MEAN = np.zeros((len(alpha_grid),len(theta_grid)))
SKEW = np.zeros_like(MEAN)
KURT = np.zeros_like(MEAN)
CV = np.zeros_like(MEAN)
for i,a in enumerate(alpha_grid):
    for j,t in enumerate(theta_grid):
        d = shape_measures(t,a)
        MEAN[i,j], SKEW[i,j], KURT[i,j], CV[i,j] = d["mean"],d["skewness"],d["kurtosis"],d["cv"]

fig, axes = plt.subplots(2,2,figsize=(10.5,7.6))
for ax,arr,title in zip(axes.flat,[MEAN,CV,SKEW,KURT],
                        ["Mean","Coefficient of variation","Skewness","Kurtosis"]):
    im=ax.imshow(arr,origin="lower",aspect="auto",
                 extent=[theta_grid.min(),theta_grid.max(),alpha_grid.min(),alpha_grid.max()])
    ax.set(title=title,xlabel=r"$\theta$",ylabel=r"$\alpha$")
    fig.colorbar(im,ax=ax,shrink=0.85)
fig.suptitle("Parameter effects on principal distributional summaries")
save_figure(fig, "figure_03_shape_measure_surfaces")

# Quantile curves
fig, ax = plt.subplots(figsize=(8.5,5.0))
pgrid = np.linspace(0.01,0.99,99)
for theta,alpha in [(0.6,0.05),(0.6,0.8),(1.5,0.2),(1.5,1.0),(3.0,0.2)]:
    ax.plot(pgrid,quantile(pgrid,theta,alpha),
            label=rf"$\theta={theta:g},\alpha={alpha:g}$")
ax.set(title="Quantile curves",xlabel="Probability $p$",ylabel="$Q(p)$")
ax.grid(alpha=0.22)
ax.legend(frameon=False)
save_figure(fig, "figure_04_quantile_curves")

# MRL and MIT
fig, axes = plt.subplots(1,2,figsize=(10.5,4.3))
for theta,alpha in [(0.5,0.2),(1.0,0.2),(2.0,0.2),(1.0,1.0)]:
    gx,mrl,mit = functional_cache.get((theta,alpha), numerical_functionals(theta,alpha,3500)[2:])
    valid_mrl = np.isfinite(mrl) & (gx <= quantile(np.array([0.99]),theta,alpha)[0])
    valid_mit = np.isfinite(mit) & (gx <= quantile(np.array([0.99]),theta,alpha)[0])
    label=rf"$\theta={theta:g},\alpha={alpha:g}$"
    axes[0].plot(gx[valid_mrl],mrl[valid_mrl],label=label)
    axes[1].plot(gx[valid_mit],mit[valid_mit],label=label)
axes[0].set(title="Mean residual life",xlabel="$t$",ylabel="$m(t)$")
axes[1].set(title="Mean inactivity time",xlabel="$t$",ylabel=r"$\widetilde{m}(t)$")
for ax in axes:
    ax.grid(alpha=0.22)
axes[0].legend(frameon=False,fontsize=8)
save_figure(fig, "figure_05_mrl_and_mit")

# ============================================================
# 3. Limiting behaviour toward Lindley as alpha -> 0
# ============================================================

limit_rows=[]
fig, axes = plt.subplots(1,2,figsize=(10.5,4.3))
for theta in [0.7,1.5,3.0]:
    alphas=np.array([1.0,0.5,0.2,0.1,0.05,0.02,0.01])
    tvs=[]; mean_diff=[]
    lind_m=shape_measures(theta,0.0)["mean"]
    xmax=quantile(np.array([0.9999]),theta,0.0)[0]
    x=np.linspace(0,xmax,5000)
    f0=pdf(x,theta,0.0)
    for a in alphas:
        tv=0.5*np.trapezoid(np.abs(pdf(x,theta,a)-f0),x)
        md=abs(shape_measures(theta,a)["mean"]-lind_m)
        tvs.append(tv); mean_diff.append(md)
        limit_rows.append({"theta":theta,"alpha":a,"TV_distance":tv,"absolute_mean_difference":md})
    axes[0].loglog(alphas,tvs,marker="o",label=rf"$\theta={theta:g}$")
    axes[1].loglog(alphas,mean_diff,marker="o",label=rf"$\theta={theta:g}$")
axes[0].set(title="Convergence in total variation",xlabel=r"$\alpha$",ylabel="TV distance")
axes[1].set(title="Convergence of the mean",xlabel=r"$\alpha$",ylabel="Absolute mean difference")
for ax in axes:
    ax.grid(alpha=0.22,which="both")
    ax.legend(frameon=False)
save_figure(fig, "figure_06_lindley_limit")
limit_df=pd.DataFrame(limit_rows)
save_table(limit_df,"table_02_lindley_limit",
           caption="Numerical convergence of QETL to the Lindley distribution as the quadratic tilt vanishes.",
           label="tab:qetl-lindley-limit")

# ============================================================
# 4. Exact random samples: visual morphology (not formula audit)
# ============================================================

sample_settings=[(0.6,0.08),(1.0,0.5),(2.0,0.2),(3.5,1.2)]
fig,axes=plt.subplots(2,2,figsize=(10.5,7.5))
sample_rows=[]
for ax,(theta,alpha) in zip(axes.flat,sample_settings):
    samp=r_qetl(N_SAMPLE_SHAPE,theta,alpha,RNG)
    xmax=np.quantile(samp,0.995)
    bins=np.linspace(0,xmax,45)
    ax.hist(samp,bins=bins,density=True,alpha=0.5,label="Exact sample")
    x=np.linspace(0,xmax,600)
    ax.plot(x,pdf(x,theta,alpha),linewidth=2,label="QETL density")
    ax.set(title=rf"$\theta={theta:g},\alpha={alpha:g}$",xlabel="$x$",ylabel="Density")
    ax.grid(alpha=0.2)
    sample_rows.append({
        "theta":theta,"alpha":alpha,"n":len(samp),
        "sample_mean":np.mean(samp),"exact_mean":shape_measures(theta,alpha)["mean"],
        "sample_variance":np.var(samp,ddof=1),"exact_variance":shape_measures(theta,alpha)["variance"],
        "sample_skewness":pd.Series(samp).skew(),
        "exact_skewness":shape_measures(theta,alpha)["skewness"]
    })
axes[0,0].legend(frameon=False)
fig.suptitle("Morphology of exact random samples under contrasting parameter settings")
save_figure(fig,"figure_07_exact_sample_morphology")
sample_df=pd.DataFrame(sample_rows)
save_table(sample_df,"table_03_sample_shape_comparison",
           caption="Sample and exact distributional summaries for large exact QETL samples.",
           label="tab:qetl-sample-shape")

# ============================================================
# 5. Maximum-likelihood estimation
# ============================================================

def negloglik_grad_hess(params, x, need_hess=False):
    theta,alpha=params
    if theta<=0 or alpha<0:
        if need_hess:
            return np.inf,np.array([np.nan,np.nan]),np.full((2,2),np.nan)
        return np.inf,np.array([np.nan,np.nan])
    m=raw_moments(theta,alpha,4)
    n=len(x)
    ll=np.sum(np.log1p(x))-theta*np.sum(x)-alpha*np.sum(x*x)-n*logZ(theta,alpha)
    grad=np.array([-np.sum(x)+n*m[1],-np.sum(x*x)+n*m[2]])
    if need_hess:
        cov=np.array([[m[2]-m[1]**2,m[3]-m[1]*m[2]],
                      [m[3]-m[1]*m[2],m[4]-m[2]**2]])
        hess=n*cov  # Hessian of negative log likelihood
        return -ll,-grad,hess
    return -ll,-grad

def fit_qetl(x, start=None):
    x=np.asarray(x)
    if start is None:
        # Lindley-inspired start + modest tilt
        mean=max(np.mean(x),1e-4)
        theta0=max(0.15,min(8.0,1.4/mean))
        start=np.array([theta0,0.15])
    fun=lambda p: negloglik_grad_hess(p,x)[0]
    jac=lambda p: negloglik_grad_hess(p,x)[1]
    res=minimize(fun,start,jac=jac,method="L-BFGS-B",
                 bounds=[(1e-5,15.0),(0.0,8.0)],
                 options={"maxiter":180,"ftol":1e-11,"gtol":1e-7})
    theta_hat,alpha_hat=res.x
    try:
        _,_,H=negloglik_grad_hess(res.x,x,need_hess=True)
        cov=np.linalg.inv(H)
        se=np.sqrt(np.diag(cov))
    except Exception:
        cov=np.full((2,2),np.nan); se=np.array([np.nan,np.nan])
    return res,se,cov

mc_settings=[(0.7,0.15),(1.5,0.5),(3.0,1.0)]
sample_sizes=[50,100,250]
records=[]
estimate_cloud={}
for theta,alpha in mc_settings:
    for n in sample_sizes:
        est=[]
        for rep in range(N_REP):
            x=r_qetl(n,theta,alpha,RNG)
            res,se,cov=fit_qetl(x,start=np.array([theta,alpha]))
            th,al=res.x
            cover_th=np.isfinite(se[0]) and (th-1.96*se[0]<=theta<=th+1.96*se[0])
            cover_al=np.isfinite(se[1]) and (al-1.96*se[1]<=alpha<=al+1.96*se[1])
            est.append((th,al,se[0],se[1],res.success,res.nit,cover_th,cover_al))
        arr=np.array(est,dtype=float)
        estimate_cloud[(theta,alpha,n)]=arr
        for j,(name,true) in enumerate([("theta",theta),("alpha",alpha)]):
            vals=arr[:,j]
            records.append({
                "theta_true":theta,"alpha_true":alpha,"n":n,"parameter":name,
                "mean_estimate":np.mean(vals),
                "bias":np.mean(vals)-true,
                "relative_bias_percent":100*(np.mean(vals)-true)/true,
                "RMSE":np.sqrt(np.mean((vals-true)**2)),
                "empirical_SD":np.std(vals,ddof=1),
                "mean_model_SE":np.nanmean(arr[:,j+2]),
                "Wald_coverage":np.mean(arr[:,j+6]),
                "convergence_rate":np.mean(arr[:,4]),
                "mean_iterations":np.mean(arr[:,5])
            })

mc_df=pd.DataFrame(records)
save_table(mc_df,"table_04_mle_monte_carlo",
           caption="Finite-sample performance of maximum-likelihood estimation under the QETL model.",
           label="tab:qetl-mle-monte-carlo")

# Bias and RMSE panels
fig,axes=plt.subplots(2,2,figsize=(10.5,7.2))
for theta,alpha in mc_settings:
    label=rf"$({theta:g},{alpha:g})$"
    for param,col in [("theta",0),("alpha",1)]:
        d=mc_df[(mc_df.theta_true==theta)&(mc_df.alpha_true==alpha)&(mc_df.parameter==param)]
        axes[0,col].plot(d["n"],d["bias"],marker="o",label=label)
        axes[1,col].plot(d["n"],d["RMSE"],marker="o",label=label)
axes[0,0].axhline(0,linewidth=1)
axes[0,1].axhline(0,linewidth=1)
axes[0,0].set(title=r"Bias of $\widehat\theta$",xlabel="$n$",ylabel="Bias")
axes[0,1].set(title=r"Bias of $\widehat\alpha$",xlabel="$n$",ylabel="Bias")
axes[1,0].set(title=r"RMSE of $\widehat\theta$",xlabel="$n$",ylabel="RMSE")
axes[1,1].set(title=r"RMSE of $\widehat\alpha$",xlabel="$n$",ylabel="RMSE")
for ax in axes.flat:
    ax.grid(alpha=0.22)
axes[0,0].legend(title=r"$(\theta,\alpha)$",frameon=False)
save_figure(fig,"figure_08_mle_bias_rmse")

# Coverage and convergence
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
for theta,alpha in mc_settings:
    label=rf"$({theta:g},{alpha:g})$"
    dth=mc_df[(mc_df.theta_true==theta)&(mc_df.alpha_true==alpha)&(mc_df.parameter=="theta")]
    dal=mc_df[(mc_df.theta_true==theta)&(mc_df.alpha_true==alpha)&(mc_df.parameter=="alpha")]
    axes[0].plot(dth.n,dth.Wald_coverage,marker="o",label=label+r", $\theta$")
    axes[0].plot(dal.n,dal.Wald_coverage,marker="s",linestyle="--",label=label+r", $\alpha$")
    axes[1].plot(dth.n,dth.convergence_rate,marker="o",label=label)
axes[0].axhline(0.95,linestyle=":",linewidth=1.4,label="Nominal 0.95")
axes[0].set(title="Wald confidence-interval coverage",xlabel="$n$",ylabel="Coverage")
axes[1].set(title="Optimizer convergence rate",xlabel="$n$",ylabel="Proportion converged")
for ax in axes:
    ax.set_ylim(0,1.03); ax.grid(alpha=0.22)
axes[0].legend(frameon=False,fontsize=7,ncol=2)
axes[1].legend(frameon=False)
save_figure(fig,"figure_09_mle_coverage_convergence")

# Estimator clouds for n=250
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for ax,(theta,alpha) in zip(axes,mc_settings):
    arr=estimate_cloud[(theta,alpha,250)]
    ax.scatter(arr[:,0],arr[:,1],s=14,alpha=0.55)
    ax.axvline(theta,linewidth=1.2)
    ax.axhline(alpha,linewidth=1.2)
    ax.set(title=rf"True $(\theta,\alpha)=({theta:g},{alpha:g})$",
           xlabel=r"$\widehat\theta$",ylabel=r"$\widehat\alpha$")
    ax.grid(alpha=0.2)
save_figure(fig,"figure_10_mle_estimator_clouds")

# ============================================================
# 6. Likelihood geometry for one representative data set
# ============================================================

theta0,alpha0,n0=1.5,0.5,150
x0=r_qetl(n0,theta0,alpha0,RNG)
res0,se0,cov0=fit_qetl(x0,start=np.array([theta0,alpha0]))
th_hat,al_hat=res0.x

th_grid=np.linspace(max(0.2,th_hat-4*se0[0]),th_hat+4*se0[0],90)
al_grid=np.linspace(max(0,al_hat-4*se0[1]),al_hat+4*se0[1],90)
LL=np.empty((len(al_grid),len(th_grid)))
for i,a in enumerate(al_grid):
    for j,t in enumerate(th_grid):
        LL[i,j]=-negloglik_grad_hess((t,a),x0)[0]
LLmax=np.nanmax(LL)
LR=2*(LLmax-LL)

fig,ax=plt.subplots(figsize=(7.2,5.3))
cs=ax.contour(th_grid,al_grid,LR,levels=[2.30,5.99,9.21])
ax.clabel(cs,inline=True,fontsize=8,fmt={2.30:"68%",5.99:"95%",9.21:"99%"})
ax.scatter([theta0],[alpha0],marker="*",s=130,label="True parameter")
ax.scatter([th_hat],[al_hat],marker="o",s=55,label="MLE")
ax.set(title="Likelihood-ratio contours for a representative sample",
       xlabel=r"$\theta$",ylabel=r"$\alpha$")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
save_figure(fig,"figure_11_likelihood_contours")

likelihood_df=pd.DataFrame([{
    "theta_true":theta0,"alpha_true":alpha0,"n":n0,
    "theta_hat":th_hat,"alpha_hat":al_hat,
    "SE_theta":se0[0],"SE_alpha":se0[1],
    "estimate_correlation":cov0[0,1]/math.sqrt(cov0[0,0]*cov0[1,1]),
    "optimizer_success":res0.success,"iterations":res0.nit
}])
save_table(likelihood_df,"table_05_likelihood_geometry",
           caption="MLE and observed likelihood geometry for the representative sample used in the contour plot.",
           label="tab:qetl-likelihood-geometry")

# ============================================================
# 7. Boundary behaviour under alpha = 0
# ============================================================

def fit_lindley_theta(x):
    # closed-form MLE from score: 2n/theta - n/(theta+1) - sum x = 0
    n=len(x); sx=np.sum(x)
    # sx*theta^2 + (sx-n)*theta - 2n = 0
    A=sx; B=sx-n; C=-2*n
    return (-B+math.sqrt(B*B-4*A*C))/(2*A)

boundary_records=[]
boundary_n=[50,100,250]
B_REP=max(220,N_REP)
for n in boundary_n:
    lrvals=[]; ahats=[]; conv=[]
    for rep in range(B_REP):
        x=r_qetl(n,1.5,0.0,RNG)
        th0=fit_lindley_theta(x)
        ll0=-negloglik_grad_hess((th0,0.0),x)[0]
        res,se,cov=fit_qetl(x,start=np.array([th0,0.03]))
        ll1=-res.fun
        lr=max(0.0,2*(ll1-ll0))
        lrvals.append(lr); ahats.append(res.x[1]); conv.append(res.success)
    lrvals=np.array(lrvals); ahats=np.array(ahats)
    # 5% critical value for 0.5*chi0^2 + 0.5*chi1^2 is chi1^2 90th percentile
    crit=chi2.ppf(0.90,1)
    boundary_records.append({
        "theta_true":1.5,"alpha_true":0.0,"n":n,"replications":B_REP,
        "mean_alpha_hat":np.mean(ahats),
        "median_alpha_hat":np.median(ahats),
        "proportion_alpha_hat_near_zero":np.mean(ahats<1e-6),
        "mean_LR":np.mean(lrvals),
        "empirical_size_mixture_5pct":np.mean(lrvals>crit),
        "convergence_rate":np.mean(conv)
    })
boundary_df=pd.DataFrame(boundary_records)
save_table(boundary_df,"table_06_boundary_simulation",
           caption="Boundary simulation under the Lindley submodel alpha=0.",
           label="tab:qetl-boundary-simulation")

fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
# use n=100 fresh/reuse? rerun one moderate sample for graph
lrplot=[]; ahplot=[]
for rep in range(B_REP):
    x=r_qetl(100,1.5,0.0,RNG)
    th0=fit_lindley_theta(x)
    ll0=-negloglik_grad_hess((th0,0.0),x)[0]
    res,se,cov=fit_qetl(x,start=np.array([th0,0.03]))
    lrplot.append(max(0,2*((-res.fun)-ll0))); ahplot.append(res.x[1])
axes[0].hist(ahplot,bins=30,density=True,alpha=0.7)
axes[0].set(title=r"Boundary estimates of $\alpha$",xlabel=r"$\widehat\alpha$",ylabel="Density")
axes[1].hist(lrplot,bins=30,density=True,alpha=0.7,label="Empirical LR")
xx=np.linspace(0.001,max(8,np.quantile(lrplot,0.99)),400)
axes[1].plot(xx,0.5*chi2.pdf(xx,1),linewidth=2,label=r"$\frac{1}{2}\chi_1^2$ continuous part")
axes[1].set(title="Likelihood-ratio statistic under the boundary null",
            xlabel="LR statistic",ylabel="Density")
for ax in axes: ax.grid(alpha=0.2)
axes[1].legend(frameon=False)
save_figure(fig,"figure_12_boundary_behaviour")

# ============================================================
# 8. Master summary and ZIP
# ============================================================

master = pd.DataFrame([
    {"component":"Distributional atlas","output":"Density, CDF, survival, hazard and cumulative-hazard curves"},
    {"component":"Shape analysis","output":"Mean, variance, CV, skewness, kurtosis, quantiles, entropy and Gini"},
    {"component":"Ageing analysis","output":"Mean residual life and mean inactivity time"},
    {"component":"Limiting model","output":"Total-variation and mean convergence toward Lindley as alpha -> 0"},
    {"component":"Exact sampling morphology","output":"Histograms against theoretical densities and moment comparisons"},
    {"component":"MLE simulation","output":"Bias, relative bias, RMSE, SD, model SE, coverage and convergence"},
    {"component":"Likelihood geometry","output":"Likelihood-ratio contours and estimator dependence"},
    {"component":"Boundary analysis","output":"alpha=0 estimates and mixture-chi-square likelihood-ratio behaviour"},
])
save_table(master,"table_00_output_inventory",
           caption="Inventory of the additional QETL simulation study.",
           label="tab:qetl-simulation-inventory")

runtime=time.time()-start
metadata={
    "seed":SEED,"fast_mode":FAST_MODE,"replications":N_REP,
    "large_sample_size":N_SAMPLE_SHAPE,"runtime_seconds":runtime,
    "model":"f(x) proportional to (1+x) exp(-theta*x-alpha*x^2), x>0"
}
(ROOT/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")

print("\n" + "="*70)
print("SIMULATION STUDY COMPLETE")
print("="*70)
print(f"Runtime: {runtime:.2f} seconds")
print(f"Figures saved: {len(list(FIG_DIR.glob('*.png')))} PNG + {len(list(FIG_DIR.glob('*.pdf')))} PDF")
print(f"Tables saved: {len(list(TAB_DIR.glob('*.csv')))} CSV + {len(list(TAB_DIR.glob('*.tex')))} LaTeX")

zip_name=Path("QETL_REMAINING_SIMULATION_RESULTS.zip")
if zip_name.exists():
    zip_name.unlink()
with zipfile.ZipFile(zip_name,"w",zipfile.ZIP_DEFLATED) as zf:
    for path in ROOT.rglob("*"):
        if path.is_file():
            zf.write(path,path.as_posix())

print(f"Created: {zip_name.resolve()}")

try:
    from google.colab import files
    files.download(str(zip_name))
except Exception:
    print("Not running in Colab. Download the ZIP from the current working directory.")
