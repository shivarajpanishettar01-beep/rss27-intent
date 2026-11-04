"""Synthetic hold-world generator: ground truth for every test. SPEC section 3. Milestone M1.

SYNTHETIC ONLY. Outputs are test oracles with known answers; they are never experimental
results and must never be saved under results/ or feed a paper figure. Every parameter below is a
simulation parameter, not a rig fact.

World model (units: s, m, rad, N):
- Stations lie on a straight 1-D vessel along base x, ``station_spacing_m`` apart. The probe's
  tool frame is R_z(``tool_yaw_rad``) relative to base; tool z (probe axis) is base z.
- Episode = approach (descend into contact) -> hold_0 -> exit -> hold_1 -> ... -> hold_{n-1} ->
  exit tail. Holds last D ~ p_D from arrival to the operator key press; the cue precedes the key
  by a reaction time R ~ LogNormal (cue = key - R, so D, not the cue offset, follows p_D).
- Every exit starts at the key press with the same tangential velocity ramp
  v(t) = v * s(t / t_acc), s(u) = 10u^3 - 15u^4 + 6u^5 (minimum-jerk shape), cruises at
  v in ``speeds_mps`` and ramps down symmetrically at the next station. LIFT exits additionally
  raise z by ``lift_height_m`` (minimum-jerk position profile) and recontact before arrival.
  Back-tracking exits go to the previous station.
- B-mode: rows = depth, cols = lateral. log-compressed tissue level + a speckle field that is a
  Gaussian process along the elevational (sweep) direction with correlation
  rho(d) = exp(-d^2 / (2 sigma^2)), sigma = ``speckle_sigma_m``, + i.i.d. electronic noise.
  A dark ellipse is the vessel. Lifted probe => uncoupled columns go dark.
- Leak mode: adequacy(t) rises from ``adequacy_base`` to 1 over ``leak_lead_s`` before each key
  press and reduces depth attenuation, so the image carries information about hold end. With the
  leak off adequacy is constant and hold frames are i.i.d. given the station.
- Wrench (tool frame): Fz = +``force_n`` (compressive normal force, positive) + noise in contact.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from usintent.data.chunks import actions_from_poses
from usintent.data.loaders import HOLD_COLUMNS, Episode
from usintent.theory.distributions import HoldDistribution, Uniform

SYNTHETIC_LABEL = "synthetic"


@dataclass(frozen=True)
class HoldWorldConfig:
    """Synthetic world parameters (NOT rig facts)."""

    hold_dist: HoldDistribution = field(default_factory=lambda: Uniform(2.0, 10.0))
    distribution_name: str = "uniform"
    n_stations: tuple[int, int] = (3, 6)             # inclusive range of holds per episode
    station_spacing_m: float = 0.05
    speeds_mps: tuple[float, ...] = (0.01, 0.02, 0.03)
    t_acc_s: float = 0.5
    fps: float = 30.0
    horizon_s: float = 1.0
    approach_s: float = 1.5
    tail_s: float = 3.0                              # motion recorded after the last key press
    reaction_lognorm: tuple[float, float] = (np.log(0.35), 0.25)   # (mu, sigma) of log R
    p_lift: float = 0.2
    p_backtrack: float = 0.15
    lift_height_m: float = 0.01
    lift_time_s: float = 0.4
    tool_yaw_rad: float = 0.0
    force_n: float = 5.0
    force_noise_n: float = 0.2
    image_shape: tuple[int, int] = (64, 64)
    render_images: bool = True
    speckle_sigma_m: float = 0.0015
    speckle_features: int = 16
    speckle_amp: float = 35.0                       # uint8 levels
    noise_amp: float = 4.0                          # i.i.d. electronic noise, uint8 levels
    tissue_level: float = 120.0
    attenuation: float = 60.0                       # level loss at full depth when adequacy = 0
    adequacy_base: float = 0.5
    leak: bool = False
    leak_lead_s: float = 1.5
    vessel_center_rc: tuple[float, float] = (0.55, 0.5)   # fractions of (rows, cols)
    vessel_radii_rc: tuple[float, float] = (0.12, 0.16)
    session_id: str = "syn-s0"
    day: str = "syn-d0"
    operator_id: str = "syn-op0"
    block_id: int = 0
    seed: int = 0


# ------------------------------------------------------------------------------- kinematics


def _s(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Minimum-jerk ramp shape on [0, 1]: 0 -> 1 with zero slope and curvature at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return np.asarray(10 * u**3 - 15 * u**4 + 6 * u**5, dtype=np.float64)


def _int_s(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """Integral of s from 0 to u (u in [0, 1]); equals 1/2 at u = 1."""
    u = np.clip(u, 0.0, 1.0)
    return np.asarray(2.5 * u**4 - 3 * u**5 + u**6, dtype=np.float64)


def sweep_profile(t: NDArray[np.float64], dist_m: float, v: float, t_acc: float
                  ) -> tuple[NDArray[np.float64], float]:
    """Distance travelled (m) at times t (s, from exit start) for one sweep; returns (x, duration).

    Ramp-up v*s(t/t_acc), cruise at v, ramp-down symmetric. If the distance is too short for a
    cruise phase the peak speed is lowered to dist/t_acc.
    """
    if dist_m <= v * t_acc:
        v = dist_m / t_acc
    t_cruise = dist_m / v - t_acc
    dur = 2 * t_acc + t_cruise
    t = np.asarray(t, dtype=np.float64)
    x = np.where(
        t <= t_acc, v * t_acc * _int_s(t / t_acc),
        np.where(t <= t_acc + t_cruise, v * t_acc / 2 + v * (t - t_acc),
                 dist_m - v * t_acc * _int_s((dur - t) / t_acc)))
    return np.clip(np.where(t >= dur, dist_m, np.where(t <= 0, 0.0, x)), 0.0, dist_m), dur


def departure_tau(threshold_m: float, v: float, t_acc: float) -> float:
    """Time (s) after the key press at which the exit displacement first exceeds threshold_m.

    Valid while the sweep is long enough to reach cruise (true for all synthetic defaults).
    """
    ramp = v * t_acc / 2
    if threshold_m <= 0:
        return 0.0
    if threshold_m < ramp:
        return float(t_acc * brentq(lambda u: v * t_acc * float(_int_s(np.array(u))) - threshold_m,
                                    0.0, 1.0, xtol=1e-12))
    return t_acc + (threshold_m - ramp) / v


# ------------------------------------------------------------------------------- B-mode


class _Speckle:
    """Speckle field: per pixel a Gaussian process along elevational position x (m)."""

    def __init__(self, shape: tuple[int, int], sigma: float, k: int, rng: np.random.Generator):
        npx = shape[0] * shape[1]
        self.shape = shape
        self.omega = rng.normal(0.0, 1.0 / sigma, (npx, k))
        self.phase = rng.uniform(0.0, 2 * np.pi, (npx, k))
        self.scale = np.sqrt(2.0 / k)

    def at(self, x: float) -> NDArray[np.float64]:
        f = self.scale * np.cos(self.omega * x + self.phase).sum(axis=1)
        return np.asarray(f.reshape(self.shape), dtype=np.float64)


def render_bmode(cfg: HoldWorldConfig, x_elev: NDArray[np.float64], coupling: NDArray[np.float64],
                 adequacy: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.uint8]:
    """Render (N, rows, cols) uint8 frames for elevational positions x (m), coupling in [0, 1]
    (fraction of columns in contact, centred) and adequacy in [0, 1]."""
    rows, cols = cfg.image_shape
    speckle = _Speckle(cfg.image_shape, cfg.speckle_sigma_m, cfg.speckle_features, rng)
    depth = (np.arange(rows) / (rows - 1))[:, None]                       # 0 near field .. 1
    rr, cc = np.mgrid[0:rows, 0:cols]
    (vr, vc), (ar, ac) = cfg.vessel_center_rc, cfg.vessel_radii_rc
    vessel = ((rr - vr * rows) / (ar * rows)) ** 2 + ((cc - vc * cols) / (ac * cols)) ** 2 <= 1.0
    colpos = np.abs(np.arange(cols) - (cols - 1) / 2) / ((cols - 1) / 2)   # 0 centre .. 1 edge
    frames = np.empty((len(x_elev), rows, cols), dtype=np.uint8)
    for k in range(len(x_elev)):
        level = cfg.tissue_level - cfg.attenuation * (1.0 - adequacy[k]) * depth
        img = level + cfg.speckle_amp * speckle.at(float(x_elev[k]))
        img = np.where(vessel, 15.0, img)
        coupled = colpos <= coupling[k] + 1e-9 if coupling[k] < 1.0 else np.ones(cols, bool)
        img = np.where(coupled[None, :], img, 5.0)
        img = img + cfg.noise_amp * rng.standard_normal(img.shape)
        frames[k] = np.clip(np.round(img), 0, 255).astype(np.uint8)
    return frames


# ------------------------------------------------------------------------------- episodes


@dataclass(frozen=True)
class _Planned:
    hold_id: int
    station: int
    next: int
    start: float
    key: float
    cue: float
    D: float
    R: float
    exit_type: str
    backtrack: bool
    v: float
    dur: float


def generate_episode(cfg: HoldWorldConfig, episode_index: int, rng: np.random.Generator,
                     first_hold_id: int = 0, truncate_at_s: float | None = None) -> Episode:
    """Generate one synthetic episode. ``truncate_at_s`` cuts it (right-censoring the open hold)."""
    n_holds = int(rng.integers(cfg.n_stations[0], cfg.n_stations[1] + 1))
    t_acc, dt = cfg.t_acc_s, 1.0 / cfg.fps
    # --- event plan
    holds: list[_Planned] = []
    station, t = 0, cfg.approach_s
    for j in range(n_holds):
        d = float(cfg.hold_dist.sample(1, rng)[0])
        r = float(np.exp(rng.normal(*cfg.reaction_lognorm)))
        r = min(r, 0.9 * d)
        start, key = t, t + d
        backtrack = station > 0 and rng.random() < cfg.p_backtrack
        nxt = station - 1 if backtrack else station + 1
        exit_type = "LIFT" if rng.random() < cfg.p_lift else "SWEEP"
        v = float(rng.choice(cfg.speeds_mps))
        _, dur = sweep_profile(np.zeros(1), cfg.station_spacing_m, v, t_acc)
        holds.append(_Planned(hold_id=first_hold_id + j, station=station, next=nxt, start=start,
                              key=key, cue=key - r, D=d, R=r, exit_type=exit_type,
                              backtrack=bool(backtrack), v=v, dur=dur))
        station, t = nxt, key + dur
    t_end = holds[-1].key + cfg.tail_s
    if truncate_at_s is not None:
        t_end = min(t_end, truncate_at_s)
    time = np.arange(0.0, t_end, dt, dtype=np.float64)
    n = len(time)
    # --- trajectories
    x = np.zeros(n)
    z_lift = np.zeros(n)
    speed = np.zeros(n)
    mode = np.full(n, "RECONTACT", dtype="<U9")
    cmd = np.full(n, "RECONTACT", dtype="<U9")
    station_idx = np.full(n, -1, dtype=np.int64)
    hold_id = np.full(n, -1, dtype=np.int64)
    adequacy = np.full(n, cfg.adequacy_base)
    app = time < cfg.approach_s
    z_lift[app] = cfg.lift_height_m * (1.0 - _s(time[app] / cfg.approach_s))
    for h in holds:
        x0 = h.station * cfg.station_spacing_m
        direction = np.sign(h.next - h.station)
        in_hold = (time >= h.start) & (time < h.key)
        x[in_hold] = x0
        station_idx[in_hold] = h.station
        hold_id[in_hold] = h.hold_id
        cmd[in_hold] = "HOLD"
        mode[(time >= h.start) & (time < h.cue)] = "HOLD"
        if cfg.leak:
            lead = (time - (h.key - cfg.leak_lead_s)) / cfg.leak_lead_s
            adequacy[in_hold] = cfg.adequacy_base + (1 - cfg.adequacy_base) * _s(lead[in_hold])
        sel = (time >= h.key) & (time < h.key + h.dur)
        tt = time[sel] - h.key
        dist, _ = sweep_profile(tt, cfg.station_spacing_m, h.v, t_acc)
        x[sel] = x0 + direction * dist
        speed[sel] = np.gradient(dist, dt) if sel.sum() > 1 else 0.0
        if h.exit_type == "LIFT":
            up = _s(tt / cfg.lift_time_s)
            down = _s((h.dur - tt) / cfg.lift_time_s)
            z_lift[sel] = cfg.lift_height_m * np.minimum(up, down)
            cmd[sel] = np.where(tt < h.dur / 2, "LIFT", "RECONTACT")
        else:
            cmd[sel] = "SWEEP"
        # scheduler mode after the cue: the next mode the protocol asks for
        mode[(time >= h.cue) & (time < h.key + h.dur)] = h.exit_type
    # Frames: base z is up; the probe points down, so R_base_tool = R_x(pi) R_z(yaw) and tool z
    # (into tissue) = -base z. Lifting the probe increases base z.
    r_tool = Rotation.from_rotvec([np.pi, 0.0, 0.0]) * Rotation.from_rotvec([0.0, 0.0, cfg.tool_yaw_rad])
    pose = np.zeros((n, 6))
    pose[:, 0] = x
    pose[:, 2] = z_lift
    pose[:, 3:6] = r_tool.as_rotvec()
    state = np.zeros((n, 12))
    state[:, :6] = pose            # joints 6..11 stay 0: synthetic world has no arm kinematics
    action = actions_from_poses(pose)
    in_contact = z_lift < 1e-5
    coupling = np.clip(1.0 - z_lift / (0.25 * cfg.lift_height_m), 0.0, 1.0)
    wrench = np.zeros((n, 6))
    wrench[:, 2] = np.where(in_contact, cfg.force_n, cfg.force_n * coupling * 0.5)
    wrench += cfg.force_noise_n * rng.standard_normal((n, 6))
    bmode = None
    if cfg.render_images:
        bmode = render_bmode(cfg, x, coupling, adequacy, rng)
    rows = []
    for h in holds:
        censored = h.key >= t_end
        if h.start >= t_end:
            break
        rows.append({
            "hold_id": h.hold_id, "episode_index": episode_index, "station_index": h.station,
            "start_time": h.start, "cue_time": h.cue if h.cue < t_end else np.nan,
            "key_time": np.nan if censored else h.key, "D": np.nan if censored else h.D,
            "reaction_time": np.nan if censored else h.R, "exit_type": h.exit_type,
            "backtrack": bool(h.backtrack), "sampled_D": h.cue - h.start,
            "distribution": cfg.distribution_name, "block_id": cfg.block_id, "censored": censored,
            "gt_exit_speed_mps": h.v, "gt_D": h.D, "gt_exit_direction": float(
                np.sign(h.next - h.station)),
        })
    holds_df = pd.DataFrame(rows, columns=[*HOLD_COLUMNS, "gt_exit_speed_mps", "gt_D",
                                           "gt_exit_direction"])
    ep = Episode(
        episode_index=episode_index, fps=cfg.fps, time=time, state=state, action=action,
        wrench=wrench, mode=mode, contact_cmd=cmd, station_index=station_idx, hold_id=hold_id,
        session_id=cfg.session_id, day=cfg.day, operator_id=cfg.operator_id, holds=holds_df,
        bmode=bmode,
        extras={"gt.adequacy": adequacy, "gt.in_contact": in_contact.astype(np.float64),
                "gt.speed_mps": np.abs(speed)},
    )
    ep.validate()
    return ep


def generate(cfg: HoldWorldConfig, n_episodes: int, sessions: int = 1, days: int = 1
             ) -> list[Episode]:
    """Generate ``n_episodes`` episodes, deterministic given ``cfg.seed``.

    Episodes are spread round-robin over ``sessions`` sessions; session j belongs to day
    j % days. Session/day ids are "syn-s<j>", "syn-d<j % days>".
    """
    rng = np.random.default_rng(cfg.seed)
    episodes, hold_id = [], 0
    for i in range(n_episodes):
        s = i % sessions
        c = replace(cfg, session_id=f"syn-s{s}", day=f"syn-d{s % days}")
        ep = generate_episode(c, i, rng, first_hold_id=hold_id)
        hold_id += len(ep.holds)
        episodes.append(ep)
    return episodes


def generate_holds(cfg: HoldWorldConfig, n_holds: int) -> list[Episode]:
    """Generate episodes until at least ``n_holds`` holds exist (images off for speed if unset)."""
    rng = np.random.default_rng(cfg.seed)
    episodes, hold_id, i = [], 0, 0
    while hold_id < n_holds:
        ep = generate_episode(cfg, i, rng, first_hold_id=hold_id)
        hold_id += len(ep.holds)
        episodes.append(ep)
        i += 1
    return episodes
