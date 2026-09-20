"""README Quick start — self-contained and executable (used by check_release.sh)."""
import numpy as np
from stamps.estimation.kriging import kriging
from stamps.stats.dependence.covariance.stcov import stcov
from stamps.stats.dependence.covariance.stcovfit import covmodelfit   # needs nlopt
from stamps.bme.BMEprobaEstimations import BMEPosteriorMoments
from stamps.bme.softconverter import probaUniform

# --- 0. Synthetic data so the example is self-contained ---
rng = np.random.default_rng(0)
ch = rng.uniform(0, 10_000, size=(150, 2))                  # hard data locations (m)
zh = 20 + 3*np.sin(ch[:, 0]/2_000) + rng.normal(0, 1.5, 150)  # hard data values
cs = rng.uniform(0, 10_000, size=(30, 2))                   # soft data locations
z_mid = 20 + 3*np.sin(cs[:, 0]/2_000)
z_low, z_high = z_mid - 2.0, z_mid + 2.0                    # interval soft data
gx, gy = np.meshgrid(np.linspace(0, 10_000, 20), np.linspace(0, 10_000, 20))
ck = np.column_stack([gx.ravel(), gy.ravel()])              # estimation grid

# --- 1. Empirical covariance and model fit ---
lagS       = np.arange(0, 8_000, 500.0)
lagS_range = np.full(len(lagS), 250.0)
cov_h, n_h, lag_h, _ = stcov(ch, None, zh, lagS, lagS_range)
ok = (n_h > 0) & ~np.isnan(cov_h)
lag_h, cov_h, n_h = lag_h[ok], cov_h[ok], n_h[ok]

covmodel  = ['nuggetC', 'sphericalC']
covparam0 = [(1.0,), (4.0, 4_000.0)]                        # initial guesses
covparam, _ = covmodelfit(lag_h.reshape(-1, 1), np.array([[0.0]]),
                          cov_h.reshape(-1, 1), n_h.reshape(-1, 1),
                          covmodel, covparam0)

# --- 2. Ordinary kriging ---
zk_mean, zk_var = kriging(ck, ch, zh, covmodel, covparam,
                          nhmax=20, dmax=8_000.0, order=0)

# --- 3. BME with interval soft data ---
softpdftype, nl, limi, probdens = probaUniform(z_low, z_high)
zs = [(softpdftype, nl[i:i+1], limi[i:i+1], probdens[i:i+1]) for i in range(len(cs))]
result = BMEPosteriorMoments(ck, ch=ch, cs=cs, zh=zh.reshape(-1, 1), zs=zs,
                             covmodel=covmodel, covparam=covparam,
                             order=0,                       # constant prior mean
                             nhmax=15, nsmax=8, dmax=np.array([[8_000.0]]))
bme_mean, bme_var = result[:, 0], result[:, 1]
print(f"kriging mean {np.nanmean(zk_mean):.2f}; BME mean {np.nanmean(bme_mean):.2f}")
