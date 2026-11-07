#!/usr/bin/env python
"""Module 4 v2: virtual sensors (redundant-neighbour regression) and the linear physics fallback.

VirtualSensorBank   per-channel ridge regression of a sensor from its most correlated healthy neighbours, used for
                    long gaps on redundant sensors. Neighbours are ranked by |correlation| once at fit time; at
                    prediction time failed channels are skipped and the k best surviving ones are used.
LinearFluxFallback  the "physics fallback" for critical total loss: the linear least-squares (Gaussian conditional
                    mean) estimate of the whole 65 x 65 psi map from whichever measurements are still available
                    (flux loops, PF coil currents, solenoid current, Ip ...), fitted on training windows.
                    It is a statistical linear reconstruction. It is NOT EFIT and does not solve the
                    Grad-Shafranov equation; it only exploits that psi is, to first order, linear in the coil
                    currents and the plasma current.

Both take plain arrays so the Kaggle / Modal harness can fit them on training windows:
    bank = fit_virtual_sensors(train_X)          # train_X: (N, T, C) or (M, C) canonical channel matrix
    fb   = fit_physics_fallback(train_feats, psi)  # train_feats: (N, C) window means, psi: (N, H, W)
"""
import numpy as np


def _rows(X):
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(-1, X.shape[-1])


class VirtualSensorBank:
    def __init__(self, mu, sd, corr, k, ridge):
        self.mu, self.sd, self.corr, self.k, self.ridge = mu, sd, corr, k, ridge
        C = len(mu)
        c = np.abs(corr.copy())
        np.fill_diagonal(c, -1.0)
        self.order = np.argsort(-c, axis=1)[:, :-1]          # neighbours by descending |corr|, self excluded
        self.r2 = np.array([self._r2(i) for i in range(C)])  # in-sample R^2 with all channels healthy

    def neighbours(self, c, failed=None, k=None):
        ok = self.order[c] if failed is None else self.order[c][~failed[self.order[c]]]
        return ok[: (k or self.k)]

    def weights(self, c, nb):
        S = self.corr
        return np.linalg.solve(S[np.ix_(nb, nb)] + self.ridge * np.eye(len(nb)), S[nb, c])

    def _r2(self, c):
        nb = self.neighbours(c)
        return float(np.clip(self.corr[nb, c] @ self.weights(c, nb), 0.0, 1.0))

    def r2_given(self, c, failed):
        """R^2 the virtual sensor for channel c would have with the channels in `failed` (C,) bool unavailable."""
        nb = self.neighbours(c, failed)
        return float(np.clip(self.corr[nb, c] @ self.weights(c, nb), 0.0, 1.0)) if len(nb) else 0.0

    def predict(self, X, c, failed=None):
        """Virtual series for channel c from the neighbours in X (T, C); NaN neighbours count as failed."""
        failed = np.zeros(X.shape[1], bool) if failed is None else failed.copy()
        failed |= ~np.isfinite(X).all(axis=0)
        failed[c] = True
        nb = self.neighbours(c, failed)
        if len(nb) == 0:
            return np.full(X.shape[0], self.mu[c])
        z = (X[:, nb] - self.mu[nb]) / self.sd[nb]
        return self.mu[c] + self.sd[c] * (z @ self.weights(c, nb))

    def repair(self, X, bad, sample_mask=None, failed=None):
        """Copy of X where channels in `bad` (C,) bool take the virtual value (only at sample_mask (T, C) if given).
        All bad channels are excluded from the neighbours of every other bad channel."""
        out = X.copy()
        failed = bad.copy() if failed is None else (failed | bad)
        for c in np.flatnonzero(bad):
            v = self.predict(X, c, failed)
            if sample_mask is None:
                out[:, c] = v
            else:
                out[sample_mask[:, c], c] = v[sample_mask[:, c]]
        return out

    def evaluate(self, X_heldout, gap_frac=0.8, extra_failed=0, seed=0):
        """Reconstruction error on clean held-out windows (N, T, C): each channel in turn loses the last `gap_frac` of
        the window (plus `extra_failed` random other channels, to mimic a multi-sensor outage). Compared with
        forward-fill (hold the last good sample) and zero-fill. RMSE in units of the channel's own sigma."""
        rng = np.random.default_rng(seed)
        N, T, C = X_heldout.shape
        g0 = max(1, int(round((1.0 - gap_frac) * T)))
        err = {"virtual": [], "ffill": [], "zero": []}
        for i in range(N):
            X = X_heldout[i]
            for c in range(C):
                failed = np.zeros(C, bool)
                if extra_failed:
                    others = np.delete(np.arange(C), c)
                    failed[rng.choice(others, size=min(extra_failed, C - 1), replace=False)] = True
                v = self.predict(X, c, failed)[g0:]
                truth = X[g0:, c]
                s = self.sd[c]
                err["virtual"].append(np.mean(((v - truth) / s) ** 2))
                err["ffill"].append(np.mean(((X[g0 - 1, c] - truth) / s) ** 2))
                err["zero"].append(np.mean((truth / s) ** 2))
        out = {k: float(np.sqrt(np.mean(v))) for k, v in err.items()}
        out["virtual_beats_ffill_frac"] = float(np.mean(np.array(err["virtual"]) < np.array(err["ffill"])))
        return out


def fit_virtual_sensors(train_windows, n_neighbors=8, ridge=1e-2):
    """Fit on clean training windows (N, T, C) or rows (M, C). NaNs are replaced by the channel mean for the fit."""
    X = _rows(train_windows)
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isfinite(X), X, mu)
    sd = X.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd
    return VirtualSensorBank(mu, sd, Z.T @ Z / len(Z), n_neighbors, ridge)


class LinearFluxFallback:
    def __init__(self, mu_x, sd_x, mu_y, Sxx, Syx, ridge, shape):
        self.mu_x, self.sd_x, self.mu_y, self.Sxx, self.Syx, self.ridge, self.shape = mu_x, sd_x, mu_y, Sxx, Syx, ridge, shape

    @staticmethod
    def features(X):
        """Window mean per channel (NaN-safe): psi barely moves in 5 ms, so one value per channel is used."""
        return np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)

    def predict(self, feats, available):
        """feats (C,), available (C,) bool -> psi (H, W). With nothing available this is the training mean map."""
        a = np.flatnonzero(available & np.isfinite(feats))
        if a.size == 0:
            return self.mu_y.reshape(self.shape).copy()
        z = (feats[a] - self.mu_x[a]) / self.sd_x[a]
        w = np.linalg.solve(self.Sxx[np.ix_(a, a)] + self.ridge * np.eye(a.size), z)
        return (self.mu_y + self.Syx[:, a] @ w).reshape(self.shape)


def fit_physics_fallback(train_feats, psi, ridge=1e-2):
    """train_feats (N, C) window-mean measurements of clean windows, psi (N, H, W) the matching maps."""
    F = np.asarray(train_feats, dtype=np.float64)
    Y = np.asarray(psi, dtype=np.float64).reshape(len(F), -1)
    mu_x = np.nanmean(F, axis=0)
    F = np.where(np.isfinite(F), F, mu_x)
    sd_x = F.std(axis=0)
    sd_x = np.where(sd_x > 1e-12, sd_x, 1.0)
    Z = (F - mu_x) / sd_x
    mu_y = Y.mean(axis=0)
    N = len(Z)
    return LinearFluxFallback(mu_x, sd_x, mu_y, Z.T @ Z / N, (Y - mu_y).T @ Z / N, ridge, np.asarray(psi).shape[1:])
