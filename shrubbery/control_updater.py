"""
control_updater.py — Sequential Bayesian bias tracking for GC control samples.

After the APPAC model is fitted (appac.py), this module processes incoming
control sample observations one at a time.  kappa and all other fitted
coefficients are held fixed; only the current episode's bias distribution is
updated.

Statistical model
-----------------
For one peak j in the current episode, after covariate correction:

    log(Y_corr[t,j] / ct_ref[j])  =  b_j + ε_j,   ε_j ~ N(0, σ_j²)

The joint prior on (b_j, σ_j²) is Normal-Inverse-Gamma (NIG):

    σ_j² ~ Inv-Gamma(α, β)
    b_j | σ_j² ~ N(μ, σ_j² / κ)

NIG is conjugate for the Gaussian likelihood, so the posterior is another NIG
after every observation — no sampling required.  The marginal posterior on b_j
is Student-t(2α, μ, √(β(1+1/κ)/α)).

Sequential update on observing y (one observation, per-peak vectorised):

    κ → κ + 1
    μ → (κ·μ + y) / (κ + 1)
    α → α + ½
    β → β + κ·(y − μ)² / (2·(κ + 1))

    (right-hand sides use old values)

Breakpoint detection: Shiryaev-Roberts statistic
-------------------------------------------------
The SR statistic accumulates evidence for a step change vs. stable process:

    R_0 = 0
    R_t = (1 + R_{t-1}) · p_change(y_t) / p_stable(y_t)

where p_stable is the predictive density under the current NIG state and
p_change is the predictive density under the fresh episode prior.  Both are
Student-t; the ratio is evaluated over all peaks and combined as a sum of
log-densities (independence approximation — valid after pressure correction).

An alarm fires when R_t > sr_threshold.  The instrument-off flag
(force_breakpoint=True) starts a new episode immediately regardless of R_t.

GUM integration
---------------
Call updater.export_to_model(model) to inject the current posterior into
model.episode_bias_var and model.episode_bias so that propagate_covariance()
uses up-to-date uncertainty.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
import numpy as np

try:
    from scipy.special import gammaln as _gammaln
    _gammaln_vec = _gammaln
except ImportError:
    import math
    _gammaln_vec = np.vectorize(math.lgamma)

from .appac import AppacModel, GeneralizedCorrectionModel


# ──────────────────────────────────────────────────────────────────────────────
# NIG state — vectorised over peaks
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _NIGState:
    """
    Normal-Inverse-Gamma sufficient statistics for all peaks simultaneously.
    All arrays have shape (n_peaks,).

    Parameterisation:
        σ² ~ Inv-Gamma(alpha, beta)
        b | σ² ~ N(mu, σ² / kappa)

    Marginal posterior on b:  t_{2·alpha}( mu,  √(beta·(1 + 1/kappa)/alpha) )
    """
    kappa: np.ndarray   # precision on the mean (pseudo observation count)
    mu:    np.ndarray   # posterior mean of bias
    alpha: np.ndarray   # shape of Inv-Gamma on variance
    beta:  np.ndarray   # rate  of Inv-Gamma on variance
    n_obs: int = 0      # real observations incorporated

    def update(self, y: np.ndarray) -> '_NIGState':
        """Incorporate one observation y (n_peaks,) — returns new state."""
        kn = self.kappa + 1.0
        mn = (self.kappa * self.mu + y) / kn
        an = self.alpha + 0.5
        bn = self.beta + self.kappa * (y - self.mu) ** 2 / (2.0 * kn)
        return _NIGState(kn, mn, an, bn, self.n_obs + 1)

    def predictive_log_density(self, y: np.ndarray) -> np.ndarray:
        """
        Per-peak log p(y | state) — marginal Student-t predictive density.

        Marginalising (b, σ²) from the NIG gives
            y ~ t_{2·alpha}( mu, scale )  with scale = √(beta·(1+1/kappa)/alpha)
        """
        nu    = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (1.0 + 1.0 / self.kappa) / self.alpha)
        z     = (y - self.mu) / scale
        return (
            _gammaln_vec(0.5 * (nu + 1.0))
            - _gammaln_vec(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - np.log(scale)
            - 0.5 * (nu + 1.0) * np.log1p(z ** 2 / nu)
        )

    @property
    def posterior_mean(self) -> np.ndarray:
        """Marginal posterior mean of bias, (n_peaks,)."""
        return self.mu.copy()

    @property
    def posterior_std(self) -> np.ndarray:
        """
        Marginal posterior std of bias (from Student-t marginal).
        Defined for alpha > 1; returns np.inf for peaks where alpha ≤ 1.
        """
        var = np.where(
            self.alpha > 1.0,
            self.beta / ((self.alpha - 1.0) * self.kappa),
            np.inf,
        )
        return np.sqrt(var)

    def copy(self) -> '_NIGState':
        return _NIGState(
            self.kappa.copy(), self.mu.copy(),
            self.alpha.copy(), self.beta.copy(),
            self.n_obs,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Episode record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeRecord:
    """Immutable summary of a closed episode."""
    episode_index:    int
    start_date:       int
    end_date:         int            # first date of the next episode
    n_obs:            int
    final_mean:       np.ndarray     # (n_peaks,) posterior mean at close
    final_var:        np.ndarray     # (n_peaks,) posterior variance at close
    breakpoint_cause: str            # "forced" | "sr_alarm"
    triggering_sr:    float = 0.0   # SR value that fired the alarm (sr_alarm only)


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class ControlUpdater:
    """
    Sequential Bayesian updater for GC control sample bias.

    kappa and all other AppacModel coefficients are held fixed.
    The current episode's bias distribution is updated as new control
    observations arrive using the Normal-Inverse-Gamma conjugate prior.

    Parameters
    ----------
    model         : fitted AppacModel from appac.fit()
    sample_name   : name used to look up ct_ref and to write back into the model
    ct_ref        : (n_peaks,) reference peak areas (episode-0 mean).
                    If None, uses model.centers[sample_name].
    prior_kappa   : NIG precision on the mean prior (default 1 — one
                    pseudo-observation; weakly informative)
    prior_mu      : prior mean of bias; scalar or (n_peaks,) (default 0)
    prior_alpha   : NIG shape for variance prior (default 3 — gives a
                    finite prior variance; needs > 1 for defined mean,
                    > 2 for defined variance)
    prior_obs_std : prior estimate of log-ratio observation noise std.
                    Sets beta_0 = prior_obs_std² · (prior_alpha − 1).
                    Default 0.002 ≈ 0.2 % in fractional peak area.
                    Use from_model() to derive this from training residuals.
    sr_threshold  : Shiryaev-Roberts alarm threshold (default 500).
                    The NIG variance estimate converges to the injection-level
                    noise (typically 0.01–0.1 %).  Day-to-day atmospheric and
                    instrument variability is 10–30× larger; a low threshold
                    therefore fires on weather-driven fluctuations rather than
                    genuine gas cylinder changes.  500 gives roughly 1 alarm
                    per year on a 5-year real-data campaign; lower values
                    produce many spurious short episodes.
    min_obs       : minimum observations in the current episode before the SR
                    statistic can fire (default 5 — avoids false alarms on the
                    very first observations where the state is still diffuse)
    """

    def __init__(
        self,
        model: AppacModel,
        sample_name: str,
        ct_ref: np.ndarray | None = None,
        prior_kappa: float = 1.0,
        prior_mu: float | np.ndarray = 0.0,
        prior_alpha: float = 3.0,
        prior_obs_std: float = 0.002,
        sr_threshold: float = 500.0,
        min_obs: int = 5,
    ):
        self.model       = model
        self.sample_name = sample_name
        self.ct_ref      = (
            np.asarray(ct_ref, dtype=float) if ct_ref is not None
            else model.centers[sample_name].copy()
        )
        n_peaks = len(self.ct_ref)

        mu0 = np.broadcast_to(np.asarray(prior_mu, dtype=float), (n_peaks,)).copy()
        k0  = np.full(n_peaks, float(prior_kappa))
        a0  = np.full(n_peaks, float(prior_alpha))
        b0  = np.full(n_peaks, prior_obs_std ** 2 * (prior_alpha - 1.0))

        self._prior       = _NIGState(k0, mu0, a0, b0, n_obs=0)
        self.state        = self._prior.copy()
        self.sr_stat      = 0.0
        self.sr_threshold = float(sr_threshold)
        self.min_obs      = int(min_obs)
        self.episode      = 0
        self._start_date: int | None = None
        self.history: list[EpisodeRecord] = []
        # Episode-0 estimation variance, kept for GUM propagation into
        # subsequent episodes (set when episode 0 closes).
        self._ep0_var: np.ndarray | None = None

    # ── Class method: data-driven prior ──────────────────────────────────────

    @classmethod
    def from_model(
        cls,
        model: AppacModel,
        training_sample,        # SampleData used during fitting
        **kwargs,
    ) -> 'ControlUpdater':
        """
        Construct a ControlUpdater with prior_obs_std estimated from the
        within-episode residuals of the training data.

        This gives a better-calibrated prior than the fixed default of 0.002.

        Parameters
        ----------
        model           : fitted AppacModel
        training_sample : the SampleData that was used to fit model
        **kwargs        : forwarded to ControlUpdater.__init__
        """
        from .appac import build_multiplier, correct

        name   = training_sample.name
        ct_ref = model.centers[name]
        M      = build_multiplier(training_sample, model)
        Y_corr = correct(training_sample.Y, M)
        y_log  = np.log(Y_corr / ct_ref)                  # (n_obs, n_peaks)

        bps    = model.breakpoints[name]
        ep_ids = np.digitize(training_sample.dates, bps)
        ss, n  = 0.0, 0
        for e in range(len(bps) + 1):
            mask = ep_ids == e
            if mask.sum() > 1:
                ss += float(y_log[mask].var(axis=0).mean()) * int(mask.sum())
                n  += int(mask.sum())
        obs_std = max(float(np.sqrt(ss / n)) if n > 0 else 0.002, 1e-4)

        return cls(model, name, prior_obs_std=obs_std, **kwargs)

    # ── Covariate correction ──────────────────────────────────────────────────

    def _covariate_correct(
        self,
        Y_obs: np.ndarray,
        covariates: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        """
        Remove the fixed covariate effects using model.kappa.

        Only the multiplicative covariate term is removed here.  The episode
        bias (what we are estimating) and the cross-sample daily factor (which
        requires concurrent multi-sample data, unavailable for a single control
        series) are intentionally not applied.
        """
        denom = 1.0
        for cv, kap in self.model.kappa.items():
            if cv not in covariates:
                warnings.warn(
                    f"ControlUpdater: covariate '{cv}' in the model is not "
                    "provided — its contribution is not removed.",
                    stacklevel=3,
                )
                continue
            val   = float(covariates[cv])
            denom = denom + kap * (val - self.model.covariate_refs[cv])
        return Y_obs / denom

    # ── Episode management ────────────────────────────────────────────────────

    def _close_episode(self, date: int, cause: str, triggering_sr: float = 0.0) -> None:
        """
        Record the current episode and start a fresh one.

        When episode 0 closes, ct_ref is calibrated to the episode-0 geometric
        mean so that all future log-ratios are expressed relative to episode 0.
        The episode-0 estimation variance is saved for GUM propagation.
        """
        abs_mean = self.state.posterior_mean   # log-ratio relative to ct_ref
        abs_var  = self.state.posterior_std ** 2

        if self.episode == 0:
            # Absorb the episode-0 offset into ct_ref.  After this, log(Y/ct_ref)
            # is approximately 0 in episode 0 and equal to the true step in later
            # episodes.  The estimation uncertainty of this reference is stored
            # separately so export_to_model() can propagate it correctly.
            self.ct_ref   = self.ct_ref * np.exp(abs_mean)
            self._ep0_var = abs_var
            rec_mean      = np.zeros_like(abs_mean)  # ep0 is the reference
            rec_var       = abs_var                   # kept for GUM
        else:
            rec_mean = abs_mean
            rec_var  = abs_var

        self.history.append(EpisodeRecord(
            episode_index    = self.episode,
            start_date       = self._start_date or 0,
            end_date         = date,
            n_obs            = self.state.n_obs,
            final_mean       = rec_mean,
            final_var        = rec_var,
            breakpoint_cause = cause,
            triggering_sr    = triggering_sr,
        ))
        self.episode    += 1
        self.state       = self._prior.copy()
        self.sr_stat     = 0.0
        self._start_date = date

    # ── Single-observation update ─────────────────────────────────────────────

    def update(
        self,
        Y_obs: np.ndarray,
        covariates: dict[str, float | np.ndarray],
        date: int,
        force_breakpoint: bool = False,
    ) -> dict:
        """
        Incorporate one control observation and update the bias posterior.

        Parameters
        ----------
        Y_obs            : (n_peaks,)         raw peak areas
        covariates       : {name: scalar}     current environmental readings,
                           e.g. {"pressure": 1003.2}
        date             : int                day number (same convention as
                           SampleData.dates)
        force_breakpoint : bool               set True when the instrument was
                           switched off and on (e.g. a maintenance log flag).
                           Closes the current episode before processing this
                           observation; the observation becomes the first of
                           the new episode.

        Returns
        -------
        dict with keys:
          "bias_mean"           (n_peaks,) — posterior mean of episode bias
          "bias_std"            (n_peaks,) — posterior std
          "sr_stat"             float      — Shiryaev-Roberts statistic
          "breakpoint_detected" bool       — True if SR > threshold
          "episode"             int        — current episode index
          "n_obs"               int        — observations in current episode
          "Y_corrected"         (n_peaks,) — covariate-corrected peak areas
          "log_ratio"           (n_peaks,) — log(Y_corr / ct_ref)
        """
        if self._start_date is None:
            self._start_date = date

        Y_obs = np.asarray(Y_obs, dtype=float)

        # Force breakpoint: close before incorporating this observation so it
        # becomes the first evidence of the new episode.
        if force_breakpoint and self.state.n_obs > 0:
            self._close_episode(date, cause="forced")

        # Covariate correction and log-ratio (sufficient statistic)
        Y_corr = self._covariate_correct(Y_obs, covariates)
        y      = np.log(Y_corr / self.ct_ref)   # (n_peaks,)

        # SR test BEFORE updating: compare "y comes from stable process" vs
        # "y comes from a fresh episode".  This way an alarm means the current
        # observation is reassigned to the new episode, not the old one.
        if self.state.n_obs >= self.min_obs:
            lp_stable = self.state.predictive_log_density(y).sum()
            lp_change = self._prior.predictive_log_density(y).sum()
            lr        = np.exp(np.clip(lp_change - lp_stable, -30.0, 30.0))
            new_sr    = min((1.0 + self.sr_stat) * float(lr), 1e9)
        else:
            new_sr = 0.0

        alarm = (self.state.n_obs >= self.min_obs and new_sr > self.sr_threshold)

        if alarm:
            self._close_episode(date, cause="sr_alarm", triggering_sr=new_sr)
            new_sr = 0.0

        # Incorporate y into the current (possibly freshly reset) episode
        self.state   = self.state.update(y)
        self.sr_stat = new_sr

        return {
            "bias_mean":           self.state.posterior_mean,
            "bias_std":            self.state.posterior_std,
            "sr_stat":             self.sr_stat,
            "breakpoint_detected": alarm,
            "episode":             self.episode,
            "n_obs":               self.state.n_obs,
            "Y_corrected":         Y_corr,
            "log_ratio":           y,
        }

    # ── Batch update ──────────────────────────────────────────────────────────

    def update_batch(
        self,
        Y_obs: np.ndarray,
        covariates: dict[str, np.ndarray],
        dates: np.ndarray,
        force_breakpoints: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Process a sorted batch of control observations.

        Parameters
        ----------
        Y_obs             : (n_obs, n_peaks)
        covariates        : {name: (n_obs,)}  one value per observation
        dates             : (n_obs,)  int, sorted ascending
        force_breakpoints : (n_obs,)  bool — True where instrument was off

        Returns
        -------
        list of per-observation result dicts (same keys as update()).
        """
        n = len(dates)
        if force_breakpoints is None:
            force_breakpoints = np.zeros(n, dtype=bool)
        results = []
        for i in range(n):
            cov_i = {cv: float(arr[i]) for cv, arr in covariates.items()}
            results.append(
                self.update(Y_obs[i], cov_i, int(dates[i]), bool(force_breakpoints[i]))
            )
        return results

    # ── GUM export ────────────────────────────────────────────────────────────

    def export_to_model(self, model: AppacModel | None = None) -> AppacModel:
        """
        Inject the current posterior bias into AppacModel.episode_bias and
        AppacModel.episode_bias_var.

        After this call, propagate_covariance() will use the updater's
        uncertainty estimates rather than the training-time estimates.

        Parameters
        ----------
        model : AppacModel to modify in-place.  If None, uses self.model.

        Returns
        -------
        The modified model.
        """
        if model is None:
            model = self.model

        n_ep  = self.episode + 1
        n_pk  = len(self.ct_ref)
        bias     = np.zeros((n_ep, n_pk))
        bias_var = np.zeros((n_ep, n_pk))

        for rec in self.history:
            e = rec.episode_index
            bias[e]     = rec.final_mean
            bias_var[e] = rec.final_var

        # Current open episode
        bias[self.episode]     = self.state.posterior_mean
        bias_var[self.episode] = self.state.posterior_std ** 2

        # Convention: bias[0] = 0 (episode 0 is the reference).
        # Episodes ≥ 1 inherit the episode-0 estimation uncertainty (GUM:
        # relative bias b[e] = b_abs[e] − b_abs[0]  →  var increases by var[ep0]).
        bias[0]     = 0.0
        bias_var[0] = 0.0
        if self._ep0_var is not None:
            bias_var[1:] += self._ep0_var

        model.episode_bias[self.sample_name]     = bias
        model.episode_bias_var[self.sample_name] = bias_var

        # Breakpoint dates are the start dates of episodes 1, 2, …
        model.breakpoints[self.sample_name] = np.array(
            [rec.end_date for rec in self.history], dtype=int
        )
        return model

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"ControlUpdater  [{self.sample_name}]  current episode {self.episode}",
            f"  n_obs in episode : {self.state.n_obs}",
            f"  bias_mean  (%)   : {self.state.posterior_mean * 100}",
            f"  bias_std   (%)   : {self.state.posterior_std  * 100}",
            f"  SR statistic     : {self.sr_stat:.2f}  (threshold={self.sr_threshold})",
        ]
        if self.history:
            lines.append("  Closed episodes:")
            for rec in self.history:
                sr_str = (f"  SR={rec.triggering_sr:.0f}"
                          if rec.breakpoint_cause == "sr_alarm" else "")
                lines.append(
                    f"    ep{rec.episode_index}: days {rec.start_date}–{rec.end_date}"
                    f"  n={rec.n_obs}  cause={rec.breakpoint_cause}{sr_str}"
                    f"  bias_mean={rec.final_mean.mean()*100:+.3f}%"
                )
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Generalised sequential updater (unknown samples / ep0-referenced)
# ──────────────────────────────────────────────────────────────────────────────

class GeneralizedUpdater:
    """
    Sequential Bayesian updater for the peak-mean drift signal.

    Mirrors ControlUpdater but operates on the same representation as
    fit_generalized_correction():
      - κ-only correction of raw areas
      - Reference: gen_model.ep0_ref[instrument]  (κ-corrected ep0 geometric mean)
      - Sufficient statistic: peak-mean log-ratio → one scalar per injection

    Because the sufficient statistic is composition-independent, this updater
    is valid for any sample measured on a known instrument, including unknown
    samples whose composition need not be declared.

    The scalar NIG tracks the current-episode mean fractional drift and its
    uncertainty.  The Shiryaev-Roberts statistic detects step changes, exactly
    as in ControlUpdater.

    Parameters
    ----------
    gen_model      : GeneralizedCorrectionModel from fit_generalized_correction()
    model          : fitted AppacModel (for kappa parameters)
    instrument     : name of the reference sample (known cylinder) whose instrument
                     carries out the measurements being tracked
    prior_kappa    : NIG precision on the mean prior (default 1)
    prior_mu       : prior mean of drift (default 0 — centred on ep0)
    prior_alpha    : NIG shape for variance prior (default 3)
    prior_obs_std  : prior estimate of injection-level log-ratio noise std.
                     Use from_generalized_model() to derive from training data.
    sr_threshold   : Shiryaev-Roberts alarm threshold (default 500).
                     See ControlUpdater docstring for calibration rationale.
    min_obs        : minimum observations before SR can fire (default 5)
    """

    def __init__(
        self,
        gen_model: GeneralizedCorrectionModel,
        model: AppacModel,
        instrument: str,
        prior_kappa:   float = 1.0,
        prior_mu:      float = 0.0,
        prior_alpha:   float = 3.0,
        prior_obs_std: float = 0.002,
        sr_threshold:  float = 500.0,
        min_obs:       int   = 5,
    ):
        if instrument not in gen_model.ep0_ref:
            raise ValueError(
                f"instrument '{instrument}' not in gen_model. "
                f"Available: {sorted(gen_model.ep0_ref)}"
            )
        self.gen_model   = gen_model
        self.model       = model
        self.instrument  = instrument
        self.ct_ref      = gen_model.ep0_ref[instrument].copy()

        # Scalar NIG: shape (1,) throughout, consistent with _NIGState API
        k0 = np.array([float(prior_kappa)])
        m0 = np.array([float(prior_mu)])
        a0 = np.array([float(prior_alpha)])
        b0 = np.array([prior_obs_std ** 2 * (prior_alpha - 1.0)])

        self._prior       = _NIGState(k0, m0, a0, b0, n_obs=0)
        self.state        = self._prior.copy()
        self.sr_stat      = 0.0
        self.sr_threshold = float(sr_threshold)
        self.min_obs      = int(min_obs)
        self.episode      = 0
        self._start_date: int | None = None
        self.history: list[EpisodeRecord] = []
        self._ep0_var: float | None = None
        # Sequential drift estimates: (date, posterior_mean) after each update
        self._date_estimates: list[tuple[int, float]] = []

    # ── Class method: data-driven prior ──────────────────────────────────────

    @classmethod
    def from_generalized_model(
        cls,
        gen_model: GeneralizedCorrectionModel,
        model: AppacModel,
        training_samples: list,
        instrument: str,
        daily_aggregate: bool = True,
        **kwargs,
    ) -> 'GeneralizedUpdater':
        """
        Construct with prior_obs_std estimated from the within-episode scatter
        of the training peak-mean log-ratios, matching the noise scale used by
        fit_generalized_correction().

        Parameters
        ----------
        training_samples : SampleData list passed to fit_generalized_correction()
        instrument       : name of the reference sample / instrument to calibrate
        daily_aggregate  : if True (default) compute obs_std from daily-mean
                           log-ratios rather than per-injection values.  A GC
                           typically makes 2–5 injections per day; daily means
                           remove within-day injection noise and give the correct
                           noise scale for the day-level SR test in update_batch.
        """
        from .appac import _covariate_correct

        s = next((x for x in training_samples if x.name == instrument), None)
        if s is None:
            raise ValueError(
                f"instrument '{instrument}' not found in training_samples."
            )

        ct_ep0 = gen_model.ep0_ref[instrument]
        covariate_names = list(model.kappa.keys())
        cov_sub = {cv: s.covariates[cv] for cv in covariate_names if cv in s.covariates}
        Y_kappa = _covariate_correct(s.Y, cov_sub, model.kappa, model.covariate_refs)

        r = np.log(Y_kappa / ct_ep0[None, :]).mean(axis=1)   # (n_obs,) per injection

        bps    = model.breakpoints.get(instrument, np.array([], dtype=int))
        ep_ids = np.digitize(s.dates, bps)
        ss, n  = 0.0, 0
        for e in range(len(bps) + 1):
            mask = ep_ids == e
            if mask.sum() < 2:
                continue
            if daily_aggregate:
                dates_ep = s.dates[mask]
                r_ep     = r[mask]
                ud = np.unique(dates_ep)
                if len(ud) < 2:
                    continue
                r_daily = np.array([r_ep[dates_ep == d].mean() for d in ud])
                ss += float(r_daily.var()) * len(ud)
                n  += len(ud)
            else:
                ss += float(r[mask].var()) * int(mask.sum())
                n  += int(mask.sum())
        obs_std = max(float(np.sqrt(ss / n)) if n > 0 else 0.002, 1e-4)

        return cls(gen_model, model, instrument, prior_obs_std=obs_std, **kwargs)

    # ── κ correction ─────────────────────────────────────────────────────────

    def _kappa_correct(
        self,
        Y_obs: np.ndarray,
        covariates: dict[str, float | np.ndarray],
    ) -> np.ndarray:
        """Apply κ correction only — identical logic to ControlUpdater._covariate_correct."""
        denom = 1.0
        for cv, kap in self.model.kappa.items():
            if cv not in covariates:
                warnings.warn(
                    f"GeneralizedUpdater: covariate '{cv}' not provided — "
                    "its contribution is not removed.",
                    stacklevel=3,
                )
                continue
            denom = denom + kap * (float(covariates[cv]) - self.model.covariate_refs[cv])
        return Y_obs / denom

    # ── Episode management ────────────────────────────────────────────────────

    def _close_episode(self, date: int, cause: str, triggering_sr: float = 0.0) -> None:
        abs_mean = float(self.state.mu[0])
        abs_var  = float(self.state.posterior_std[0] ** 2)

        if self.episode == 0:
            # Absorb ep0 offset into ct_ref so future log-ratios are centred.
            # This mirrors ControlUpdater._close_episode for ep0.
            self.ct_ref   = self.ct_ref * np.exp(abs_mean)
            self._ep0_var = abs_var
            rec_mean      = np.zeros(1)
            rec_var       = np.array([abs_var])
        else:
            rec_mean = np.array([abs_mean])
            rec_var  = np.array([abs_var])

        self.history.append(EpisodeRecord(
            episode_index    = self.episode,
            start_date       = self._start_date or 0,
            end_date         = date,
            n_obs            = self.state.n_obs,
            final_mean       = rec_mean,
            final_var        = rec_var,
            breakpoint_cause = cause,
            triggering_sr    = triggering_sr,
        ))
        self.episode    += 1
        self.state       = self._prior.copy()
        self.sr_stat     = 0.0
        self._start_date = date

    # ── Single-observation update ─────────────────────────────────────────────

    def update(
        self,
        Y_obs: np.ndarray,
        covariates: dict[str, float | np.ndarray],
        date: int,
        force_breakpoint: bool = False,
    ) -> dict:
        """
        Incorporate one observation and update the scalar drift posterior.

        Parameters
        ----------
        Y_obs            : (n_peaks,)  raw peak areas of any sample on this instrument
        covariates       : {name: scalar}  current environmental readings
        date             : int  day number (same convention as SampleData.dates)
        force_breakpoint : bool  close current episode before processing this
                           observation (e.g. maintenance log flag)

        Returns
        -------
        dict with keys:
          "drift_mean"          float      — posterior mean of peak-mean log-ratio
          "drift_std"           float      — posterior std
          "sr_stat"             float      — Shiryaev-Roberts statistic
          "breakpoint_detected" bool
          "episode"             int        — current episode index
          "n_obs"               int        — observations in current episode
          "Y_kappa"             (n_peaks,) — κ-corrected peak areas
          "peak_mean_log_ratio" float      — log(Y_κ / ep0_ref).mean()
        """
        if self._start_date is None:
            self._start_date = date

        Y_obs = np.asarray(Y_obs, dtype=float)

        if force_breakpoint and self.state.n_obs > 0:
            self._close_episode(date, cause="forced")

        Y_kap  = self._kappa_correct(Y_obs, covariates)
        y_mean = float(np.log(Y_kap / self.ct_ref).mean())
        y      = np.array([y_mean])   # (1,) for _NIGState

        # SR test before update (same logic as ControlUpdater)
        if self.state.n_obs >= self.min_obs:
            lp_stable = self.state.predictive_log_density(y).sum()
            lp_change = self._prior.predictive_log_density(y).sum()
            lr        = np.exp(np.clip(lp_change - lp_stable, -30.0, 30.0))
            new_sr    = min((1.0 + self.sr_stat) * float(lr), 1e9)
        else:
            new_sr = 0.0

        alarm = (self.state.n_obs >= self.min_obs and new_sr > self.sr_threshold)
        if alarm:
            self._close_episode(date, cause="sr_alarm", triggering_sr=new_sr)
            new_sr = 0.0

        self.state   = self.state.update(y)
        self.sr_stat = new_sr
        self._date_estimates.append((date, float(self.state.mu[0])))

        return {
            "drift_mean":          float(self.state.mu[0]),
            "drift_std":           float(self.state.posterior_std[0]),
            "sr_stat":             self.sr_stat,
            "breakpoint_detected": alarm,
            "episode":             self.episode,
            "n_obs":               self.state.n_obs,
            "Y_kappa":             Y_kap,
            "peak_mean_log_ratio": y_mean,
        }

    # ── Batch update ──────────────────────────────────────────────────────────

    def update_batch(
        self,
        Y_obs: np.ndarray,
        covariates: dict[str, np.ndarray],
        dates: np.ndarray,
        force_breakpoints: np.ndarray | None = None,
        daily_aggregate: bool = True,
    ) -> list[dict]:
        """
        Process a sorted batch — same signature as ControlUpdater.update_batch.

        Parameters
        ----------
        Y_obs             : (n_obs, n_peaks)
        covariates        : {name: (n_obs,)}
        dates             : (n_obs,)  int, sorted ascending
        force_breakpoints : (n_obs,)  bool
        daily_aggregate   : if True (default) group injections by date, average
                            peak areas and covariates per day, and feed one
                            SR update per day rather than per injection.  All
                            injections on the same day receive the same result
                            dict.  This matches the noise scale used by
                            from_generalized_model(daily_aggregate=True) and
                            ensures the SR detects day-level changes rather
                            than within-day injection noise.
        """
        n = len(dates)
        if force_breakpoints is None:
            force_breakpoints = np.zeros(n, dtype=bool)

        if daily_aggregate:
            unique_dates = np.unique(dates)
            day_result: dict[int, dict] = {}
            for d in unique_dates:
                idx       = dates == d
                Y_day     = Y_obs[idx].mean(axis=0)
                cov_day   = {cv: float(covariates[cv][idx].mean()) for cv in covariates}
                force_bp  = bool(force_breakpoints[idx].any())
                day_result[int(d)] = self.update(Y_day, cov_day, int(d), force_bp)
            return [day_result[int(d)] for d in dates]

        results = []
        for i in range(n):
            cov_i = {cv: float(arr[i]) for cv, arr in covariates.items()}
            results.append(
                self.update(Y_obs[i], cov_i, int(dates[i]), bool(force_breakpoints[i]))
            )
        return results

    # ── Extend the generalised model with sequential estimates ────────────────

    def export_to_generalized(
        self,
        gen_model: GeneralizedCorrectionModel,
    ) -> GeneralizedCorrectionModel:
        """
        Extend gen_model with sequential drift estimates accumulated by this updater.

        Daily-averaged posterior means are inserted into pc2_scores for dates
        not already present in gen_model.dates_global.  The raw log-ratio is
        divided by pc2_loading[instrument] to convert back to PCA-score units,
        consistent with how build_multiplier_unknown reads the scores.

        PC1 scores for new dates are set to NaN — the correlated factor cannot
        be estimated from a single instrument without concurrent reference data.

        Returns a new GeneralizedCorrectionModel (original is not modified).
        """
        from dataclasses import replace as _replace

        if not self._date_estimates:
            return gen_model

        est_dates = np.array([d for d, _ in self._date_estimates], dtype=int)
        est_vals  = np.array([v for _, v in self._date_estimates])

        unique_est  = np.unique(est_dates)
        daily_means = np.array([est_vals[est_dates == d].mean() for d in unique_est])

        # Convert log-ratio units → PC2-score units (invert the loading projection)
        loading = gen_model.pc2_loading.get(self.instrument, 0.0)
        pc2_new = daily_means / loading if abs(loading) > 1e-12 else np.zeros_like(daily_means)

        new_dates = unique_est[~np.isin(unique_est, gen_model.dates_global)]
        if len(new_dates) == 0:
            return gen_model

        all_dates = np.sort(np.concatenate([gen_model.dates_global, new_dates]))
        n_total   = len(all_dates)

        pc1_ext = np.full(n_total, np.nan)
        pc2_ext = np.full(n_total, np.nan)

        old_idx = np.searchsorted(all_dates, gen_model.dates_global)
        pc1_ext[old_idx] = gen_model.pc1_scores
        pc2_ext[old_idx] = gen_model.pc2_scores

        new_idx = np.searchsorted(all_dates, new_dates)
        pc2_ext[new_idx] = pc2_new[~np.isin(unique_est, gen_model.dates_global)]

        return _replace(
            gen_model,
            dates_global=all_dates,
            pc1_scores=pc1_ext,
            pc2_scores=pc2_ext,
        )

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"GeneralizedUpdater  [{self.instrument}]  current episode {self.episode}",
            f"  n_obs in episode   : {self.state.n_obs}",
            f"  drift_mean (%)     : {self.state.mu[0] * 100:+.4f}",
            f"  drift_std  (%)     : {self.state.posterior_std[0] * 100:.4f}",
            f"  SR statistic       : {self.sr_stat:.2f}  (threshold={self.sr_threshold})",
        ]
        if self.history:
            lines.append("  Closed episodes:")
            for rec in self.history:
                sr_str = (f"  SR={rec.triggering_sr:.0f}"
                          if rec.breakpoint_cause == "sr_alarm" else "")
                lines.append(
                    f"    ep{rec.episode_index}: days {rec.start_date}–{rec.end_date}"
                    f"  n={rec.n_obs}  cause={rec.breakpoint_cause}{sr_str}"
                    f"  drift_mean={rec.final_mean[0]*100:+.3f}%"
                )
        return "\n".join(lines)
