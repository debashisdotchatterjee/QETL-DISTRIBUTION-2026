
import math, json, zipfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize, brentq
from scipy.special import erfcx, gammaln
from scipy.stats import expon, weibull_min, gamma as gamma_dist, lognorm, gompertz, gengamma, chi2
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED=20260728
FAST_MODE=True
ROBUSTNESS_CHECK=False
B=120 if FAST_MODE else 1000
RNG=np.random.default_rng(SEED)

ROOT=Path("QETL_WINDSHIELDS_FAST_APPLICATION")
FIG=ROOT/"figures"; TAB=ROOT/"tables"; DAT=ROOT/"data"
for d in (ROOT,FIG,TAB,DAT): d.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({
    "figure.figsize":(8,5),"figure.dpi":110,"savefig.dpi":220,"font.size":10,
    "axes.titlesize":12,"axes.spines.top":False,"axes.spines.right":False,
    "figure.constrained_layout.use":True
})

x=np.array([
0.046,1.436,2.592,0.140,1.492,2.600,0.150,1.580,2.670,
0.248,1.719,2.717,0.280,1.794,2.819,0.313,1.915,2.820,
0.389,1.920,2.878,0.487,1.963,2.950,0.622,1.978,3.003,
0.900,2.053,3.102,0.952,2.065,3.304,0.996,2.117,3.483,
1.003,2.137,3.500,1.010,2.141,3.622,1.085,2.163,3.665,
1.092,2.183,3.695,1.152,2.240,4.015,1.183,2.341,4.628,
1.244,2.435,4.806,1.249,2.464,4.881,1.262,2.543,5.140],float)

assert len(x)==63 and np.all(x>0)
pd.DataFrame({"service_time":x}).to_csv(DAT/"aircraft_windshield_service_times.csv",index=False)

scale=x.mean(); y=x/scale; n=len(y)

def J0J1(theta,alpha):
    if alpha<=1e-12: return 1/theta,1/theta**2
    a=math.sqrt(alpha); z=theta/(2*a)
    J0=math.sqrt(math.pi)*erfcx(z)/(2*a)
    J1=(1-theta*J0)/(2*alpha)
    return J0,J1

def Z(theta,alpha):
    j0,j1=J0J1(theta,alpha); return j0+j1

def qlogpdf(z,theta,alpha):
    z=np.asarray(z,float)
    if theta<=0 or alpha<0: return np.full_like(z,-np.inf)
    return np.log1p(z)-theta*z-alpha*z*z-math.log(Z(theta,alpha))

def qpdf(z,theta,alpha): return np.exp(qlogpdf(z,theta,alpha))

def qsurv(z,theta,alpha):
    z=np.asarray(z,float); out=np.ones_like(z); m=z>0; zz=z[m]
    if alpha<=1e-12:
        out[m]=(theta+1+theta*zz)*np.exp(-theta*zz)/(theta+1)
    else:
        a=math.sqrt(alpha)
        A=math.sqrt(math.pi)/(2*a)*np.exp(-alpha*zz*zz-theta*zz)*erfcx(a*zz+theta/(2*a))
        B=(np.exp(-theta*zz-alpha*zz*zz)-theta*A)/(2*alpha)
        out[m]=(A+B)/Z(theta,alpha)
    return np.clip(out,0,1)

def qcdf(z,t,a): return 1-qsurv(z,t,a)

def qhaz(z,t,a):
    z=np.asarray(z,float); S=qsurv(z,t,a)
    return np.divide(qpdf(z,t,a),S,out=np.full_like(z,np.nan),where=S>1e-14)

def qppf(p,t,a):
    p=np.asarray(p,float); out=np.empty_like(p)
    for ind,pp in np.ndenumerate(p):
        hi=max(2,2/t)
        while qcdf(np.array([hi]),t,a)[0]<pp: hi*=2
        out[ind]=brentq(lambda z:qcdf(np.array([z]),t,a)[0]-pp,0,hi)
    return out

def lindley_mle(z):
    s=z.sum(); nn=len(z)
    return (-(s-nn)+math.sqrt((s-nn)**2+8*nn*s))/(2*s)

def fit_qetl(z):
    tL=lindley_mle(z); ll0=float(qlogpdf(z,tL,0).sum())
    best={"theta":tL,"alpha":0.0,"logLik":ll0,"location":"boundary"}
    starts=[(tL,.05),(max(.05,.7*tL),.25),(max(.05,.4*tL),.75)]
    for st in starts:
        def nll(v):
            t,a=v
            if t<=0 or a<0: return 1e100
            ll=qlogpdf(z,t,a).sum()
            return -ll if np.isfinite(ll) else 1e100
        res=minimize(nll,np.array(st),method="L-BFGS-B",
                     bounds=[(1e-7,20),(0,20)],
                     options={"maxiter":250,"ftol":1e-11})
        if np.isfinite(res.fun):
            t,a=res.x; ll=-res.fun
            if ll>best["logLik"]:
                best={"theta":float(t),"alpha":float(a),"logLik":float(ll),"location":"interior"}
    best["lindley_logLik"]=ll0; best["gain"]=best["logLik"]-ll0
    return best

def edf_stats(z,cdf):
    zs=np.sort(z); nn=len(zs); F=np.clip(cdf(zs),1e-12,1-1e-12); i=np.arange(1,nn+1)
    ks=max(np.max(i/nn-F),np.max(F-(i-1)/nn))
    cvm=1/(12*nn)+np.sum((F-(2*i-1)/(2*nn))**2)
    ad=-nn-np.mean((2*i-1)*(np.log(F)+np.log(1-F[::-1])))
    return ks,cvm,ad

def IC(ll,k,nn):
    aic=-2*ll+2*k; aicc=aic+2*k*(k+1)/(nn-k-1); bic=-2*ll+k*np.log(nn)
    return aic,aicc,bic

fits=[]
def add(name,k,ll_y,cdf,pdf,ppf,haz,params):
    ll=ll_y-n*np.log(scale); ks,cvm,ad=edf_stats(y,cdf); aic,aicc,bic=IC(ll,k,n)
    fits.append({"model":name,"k":k,"logLik":ll,"AIC":aic,"AICc":aicc,"BIC":bic,
                 "KS":ks,"CvM":cvm,"AD":ad,"cdf":cdf,"pdf":pdf,"ppf":ppf,"haz":haz,"params":params})

sc=y.mean()
add("Exponential",1,expon.logpdf(y,scale=sc).sum(),
    lambda z:expon.cdf(z,scale=sc),lambda z:expon.pdf(z,scale=sc),
    lambda p:expon.ppf(p,scale=sc),lambda z:np.full_like(np.asarray(z,float),1/sc),
    {"scale_original":sc*scale})

tL=lindley_mle(y); llL=qlogpdf(y,tL,0).sum()
add("Lindley",1,llL,lambda z:qcdf(z,tL,0),lambda z:qpdf(z,tL,0),
    lambda p:qppf(p,tL,0),lambda z:qhaz(z,tL,0),
    {"theta_scaled":tL,"theta_original":tL/scale})

fq=fit_qetl(y); tQ,aQ=fq["theta"],fq["alpha"]
add("QETL",2,fq["logLik"],lambda z:qcdf(z,tQ,aQ),lambda z:qpdf(z,tQ,aQ),
    lambda p:qppf(p,tQ,aQ),lambda z:qhaz(z,tQ,aQ),
    {"theta_scaled":tQ,"alpha_scaled":aQ,"theta_original":tQ/scale,
     "alpha_original":aQ/scale**2,"loglik_gain_over_Lindley":fq["gain"]})

for name,dist in [("Weibull",weibull_min),("Gamma",gamma_dist)]:
    pars=dist.fit(y,floc=0); ll=dist.logpdf(y,*pars).sum()
    add(name,2,ll,lambda z,d=dist,p=pars:d.cdf(z,*p),lambda z,d=dist,p=pars:d.pdf(z,*p),
        lambda u,d=dist,p=pars:d.ppf(u,*p),
        lambda z,d=dist,p=pars:np.divide(d.pdf(z,*p),d.sf(z,*p),
            out=np.full_like(np.asarray(z,float),np.nan),where=d.sf(z,*p)>1e-14),
        {"scipy_parameters":tuple(float(v) for v in pars)})

if ROBUSTNESS_CHECK:
    for name,dist,k in [("Lognormal",lognorm,2),("Gompertz",gompertz,2),("Generalized gamma",gengamma,3)]:
        pars=dist.fit(y,floc=0); ll=dist.logpdf(y,*pars).sum()
        add(name,k,ll,lambda z,d=dist,p=pars:d.cdf(z,*p),lambda z,d=dist,p=pars:d.pdf(z,*p),
            lambda u,d=dist,p=pars:d.ppf(u,*p),
            lambda z,d=dist,p=pars:np.divide(d.pdf(z,*p),d.sf(z,*p),
                out=np.full_like(np.asarray(z,float),np.nan),where=d.sf(z,*p)>1e-14),
            {"scipy_parameters":tuple(float(v) for v in pars)})

fm={f["model"]:f for f in fits}

summary=pd.DataFrame([{"n":n,"minimum":x.min(),"Q1":np.quantile(x,.25),"median":np.median(x),
"mean":x.mean(),"Q3":np.quantile(x,.75),"maximum":x.max(),"SD":x.std(ddof=1),
"CV":x.std(ddof=1)/x.mean(),"skewness":pd.Series(x).skew(),"kurtosis":pd.Series(x).kurt()+3}])
display(Markdown("## Data summary")); display(summary)
summary.to_csv(TAB/"table_01_data_summary.csv",index=False)

comp=pd.DataFrame([{k:v for k,v in f.items() if k not in {"cdf","pdf","ppf","haz","params"}} for f in fits]).sort_values("AICc").reset_index(drop=True)
comp["Delta_AICc"]=comp.AICc-comp.AICc.min()
comp["Akaike_weight"]=np.exp(-.5*comp.Delta_AICc); comp["Akaike_weight"]/=comp.Akaike_weight.sum()
comp["rank"]=np.arange(1,len(comp)+1)
display(Markdown("## Main model comparison")); display(comp)
comp.to_csv(TAB/"table_02_model_comparison.csv",index=False)
comp.to_latex(TAB/"table_02_model_comparison.tex",index=False,float_format=lambda z:f"{z:.6f}",
              caption="Model comparison for aircraft-windshield service times.",
              label="tab:windshield-model-comparison")

params=pd.DataFrame([{"model":f["model"],**{k:str(v) for k,v in f["params"].items()}} for f in fits])
display(Markdown("## Parameter estimates")); display(params)
params.to_csv(TAB/"table_03_parameter_estimates.csv",index=False)

LR_obs=2*(fq["logLik"]-llL)
mixture_p=0.5*chi2.sf(LR_obs,1) if LR_obs>0 else 1.0

def r_lindley(size,theta):
    mix=RNG.random(size)<theta/(theta+1); z=np.empty(size)
    z[mix]=RNG.exponential(1/theta,mix.sum()); z[~mix]=RNG.gamma(2,1/theta,(~mix).sum())
    return z

LRb=np.empty(B)
for b in range(B):
    zb=r_lindley(n,tL); tb=lindley_mle(zb); ll0b=qlogpdf(zb,tb,0).sum(); fqb=fit_qetl(zb)
    LRb[b]=max(0,2*(fqb["logLik"]-ll0b))
boot_p=(1+np.sum(LRb>=LR_obs))/(B+1)

lrt=pd.DataFrame([{"LR_statistic":LR_obs,"mixture_chi_square_p":mixture_p,
                   "bootstrap_p":boot_p,"bootstrap_replications":B}])
display(Markdown("## Lindley versus QETL boundary test")); display(lrt)
lrt.to_csv(TAB/"table_04_boundary_test.csv",index=False)

def savefig(fig,name):
    fig.savefig(FIG/f"{name}.png",bbox_inches="tight")
    fig.savefig(FIG/f"{name}.pdf",bbox_inches="tight")
    plt.show(); plt.close(fig)

grid=np.linspace(1e-6,x.max()*1.08,650); gs=grid/scale; xs=np.sort(x)
emp=np.arange(1,n+1)/n; p=(np.arange(1,n+1)-.5)/n; models=list(comp.model)

fig,ax=plt.subplots(); ax.hist(x,bins="fd",density=True,alpha=.4,label="Observed")
for m in models: ax.plot(grid,fm[m]["pdf"](gs)/scale,lw=1.8,label=m)
ax.set(title="Fitted densities",xlabel="Service time",ylabel="Density"); ax.grid(alpha=.2); ax.legend(frameon=False)
savefig(fig,"figure_01_density")

fig,axes=plt.subplots(1,2,figsize=(10.5,4.3)); axes[0].step(xs,emp,where="post",label="Empirical"); axes[1].step(xs,1-emp,where="post",label="Empirical")
for m in models:
    F=fm[m]["cdf"](gs); axes[0].plot(grid,F,label=m); axes[1].plot(grid,1-F,label=m)
axes[0].set(title="CDF comparison",xlabel="Service time",ylabel="$F(x)$"); axes[1].set(title="Survival comparison",xlabel="Service time",ylabel="$S(x)$")
for ax in axes: ax.grid(alpha=.2)
axes[0].legend(frameon=False,fontsize=8)
savefig(fig,"figure_02_cdf_survival")

fig,ax=plt.subplots()
for m in models: ax.plot(grid,fm[m]["haz"](gs)/scale,lw=1.8,label=m)
ax.set(title="Fitted hazard functions",xlabel="Service time",ylabel="Hazard"); ax.grid(alpha=.2); ax.legend(frameon=False)
savefig(fig,"figure_03_hazard")

fig,axes=plt.subplots(1,2,figsize=(10.5,4.3)); axes[0].plot([0,1],[0,1],"--"); axes[1].plot([xs.min(),xs.max()],[xs.min(),xs.max()],"--")
for m in models:
    axes[0].plot(p,fm[m]["cdf"](xs/scale),marker="o",ms=2.4,label=m)
    axes[1].plot(fm[m]["ppf"](p)*scale,xs,marker="o",ms=2.4,label=m)
axes[0].set(title="P--P plot",xlabel="Empirical probability",ylabel="Fitted probability")
axes[1].set(title="Q--Q plot",xlabel="Fitted quantiles",ylabel="Observed quantiles")
for ax in axes: ax.grid(alpha=.2)
axes[0].legend(frameon=False,fontsize=8)
savefig(fig,"figure_04_pp_qq")

sp=np.diff(np.r_[0,xs]); ttt=np.cumsum((n-np.arange(n))*sp); phi=ttt/ttt[-1]; u=np.arange(1,n+1)/n
fig,ax=plt.subplots(figsize=(6.2,5)); ax.plot(u,phi,lw=2,label="Empirical scaled TTT"); ax.plot([0,1],[0,1],"--",label="Exponential reference")
ax.set(title="Scaled total-time-on-test transform",xlabel="$i/n$",ylabel="$T(i/n)$"); ax.grid(alpha=.2); ax.legend(frameon=False)
savefig(fig,"figure_05_ttt")

fig,ax=plt.subplots(figsize=(7.2,4.8)); dd=comp.sort_values("Delta_AICc"); ax.barh(dd.model,dd.Delta_AICc); ax.invert_yaxis(); ax.axvline(2,ls="--",lw=1.2)
ax.set(title="Relative AICc support",xlabel=r"$\Delta$AICc",ylabel="Model"); ax.grid(axis="x",alpha=.2)
savefig(fig,"figure_06_delta_aicc")

fig,ax=plt.subplots(figsize=(7.2,4.8)); ax.hist(LRb,bins=22,density=True,alpha=.45,label="Bootstrap null"); ax.axvline(LR_obs,ls="--",lw=2,label=f"Observed LR={LR_obs:.2f}")
ax.set(title="Boundary likelihood-ratio calibration",xlabel="Likelihood-ratio statistic",ylabel="Density"); ax.grid(alpha=.2); ax.legend(frameon=False)
savefig(fig,"figure_07_boundary_bootstrap")

qrow=comp[comp.model=="QETL"].iloc[0]
key=pd.DataFrame([{"dataset":"Aircraft windshield service times","n":n,"best_main_model":comp.iloc[0].model,
                   "QETL_rank":int(qrow["rank"]),"QETL_Delta_AICc":qrow["Delta_AICc"],
                   "QETL_Akaike_weight":qrow["Akaike_weight"],"QETL_loglik_gain_over_Lindley":fq["gain"],
                   "LR_statistic":LR_obs,"mixture_p":mixture_p,"bootstrap_p":boot_p,"QETL_alpha_scaled":aQ}])
display(Markdown("## Key findings")); display(key)
key.to_csv(TAB/"table_05_key_findings.csv",index=False)

(ROOT/"metadata.json").write_text(json.dumps({
"seed":SEED,"fast_mode":FAST_MODE,"bootstrap_replications":B,"robustness_check":ROBUSTNESS_CHECK,
"main_comparators":["Exponential","Lindley","Weibull","Gamma","QETL"],
"note":"Gompertz, lognormal, and generalized gamma can be included by setting ROBUSTNESS_CHECK=True."
},indent=2))

out=Path("QETL_WINDSHIELDS_FAST_RESULTS.zip")
if out.exists(): out.unlink()
with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
    for pth in ROOT.rglob("*"):
        if pth.is_file(): z.write(pth,pth.as_posix())

print("\nFAST EMPIRICAL ANALYSIS COMPLETE")
print("Best main model:",comp.iloc[0].model)
print(f"QETL vs Lindley LR={LR_obs:.4f}")
print(f"Mixture p={mixture_p:.6g}; bootstrap p={boot_p:.6g}")
print("Created:",out.resolve())

try:
    from google.colab import files
    files.download(str(out))
except Exception:
    pass
