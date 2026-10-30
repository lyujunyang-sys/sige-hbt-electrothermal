# -*- coding: utf-8 -*-
"""
Bayesian optimization with a Gaussian-process surrogate and the Expected-Improvement
acquisition function -- implemented directly from the math in Section 4 of the thesis
(rather than relying on scikit-optimize, which is not maintained for Python 3.14).

  surrogate:  f(x) ~ GP(0, k(x,x'))  with k = Const * Matern(nu=5/2) + White(noise)
  acquisition (minimisation, Eq. on p.27):
        Z   = (f_best - mu(x) - xi) / sigma(x)
        EI  = (f_best - mu(x) - xi) Phi(Z) + sigma(x) phi(Z)
  next point: x_{t+1} = argmax_x EI(x)

The search space supports per-dimension log scaling (needed for parameters such as the
saturation current I_S0 ~ 1e-16 that span many orders of magnitude).
"""
import warnings
import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, Matern, WhiteKernel)


class Param:
    def __init__(self, name, low, high, log=False):
        self.name, self.low, self.high, self.log = name, low, high, log
        self._lo = np.log10(low) if log else low
        self._hi = np.log10(high) if log else high

    def to_unit(self, v):
        x = np.log10(v) if self.log else v
        return (x - self._lo) / (self._hi - self._lo)

    def from_unit(self, u):
        x = self._lo + u * (self._hi - self._lo)
        return 10.0 ** x if self.log else x


class Space:
    def __init__(self, params):
        self.params = params
        self.d = len(params)

    def to_unit(self, vec):
        return np.array([p.to_unit(v) for p, v in zip(self.params, vec)])

    def from_unit(self, u):
        return np.array([p.from_unit(ui) for p, ui in zip(self.params, u)])

    def sample_unit(self, n, rng):
        # Latin-hypercube sampling in the unit cube
        u = np.zeros((n, self.d))
        for j in range(self.d):
            cut = (np.arange(n) + rng.random(n)) / n
            u[:, j] = rng.permutation(cut)
        return u


class BayesianOptimizer:
    def __init__(self, space, xi=0.01, seed=0):
        self.space = space
        self.xi = xi
        self.rng = np.random.default_rng(seed)
        d = space.d
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2),
                           nu=2.5)
                  + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e1)))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=4,
                                           random_state=int(self.rng.integers(1e9)))
        self.Xu = []     # evaluated points (unit cube)
        self.y = []      # objective values

    def _ei(self, Xu):
        mu, sigma = self.gp.predict(Xu, return_std=True)
        sigma = np.maximum(sigma, 1e-12)
        f_best = np.min(self.y)
        imp = f_best - mu - self.xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-11] = 0.0
        return ei

    def _propose(self, n_cand=6000):
        cand = self.rng.random((n_cand, self.space.d))
        ei = self._ei(cand)
        return cand[np.argmax(ei)]

    def run(self, objective, n_init=8, n_iter=40, verbose=True):
        """objective: callable(real_param_vector) -> float (to MINIMISE)."""
        # initial design
        for u in self.space.sample_unit(n_init, self.rng):
            self.Xu.append(u)
            self.y.append(objective(self.space.from_unit(u)))
        # BO loop
        for t in range(n_iter):
            self.gp.fit(np.array(self.Xu), np.array(self.y))
            u_next = self._propose()
            y_next = objective(self.space.from_unit(u_next))
            self.Xu.append(u_next); self.y.append(y_next)
            if verbose:
                best = np.min(self.y)
                print("  iter %3d/%d  f=%.5e  best=%.5e"
                      % (t + 1, n_iter, y_next, best))
        i = int(np.argmin(self.y))
        return self.space.from_unit(self.Xu[i]), self.y[i]

    def history_best(self):
        return np.minimum.accumulate(self.y)


if __name__ == "__main__":
    # thesis 3D benchmark: f = sin(3 x1) + cos(5 x2) + (x3-1.5)^2
    # true global minimum f=-2 at x1=1.5708, x2=0.6283, x3=1.5
    def f(v):
        x1, x2, x3 = v
        return np.sin(3 * x1) + np.cos(5 * x2) + (x3 - 1.5) ** 2

    sp = Space([Param("x1", 0.0, 2.0), Param("x2", 0.0, 2.0), Param("x3", 0.0, 3.0)])
    bo = BayesianOptimizer(sp, seed=1)
    best_x, best_f = bo.run(f, n_init=8, n_iter=35, verbose=False)
    print("BO best: x=%s  f=%.5f   (true min ~ -2.0)" %
          (np.round(best_x, 4), best_f))
