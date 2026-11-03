# SPEC.md — technical specification

This document defines what every module must compute. Section numbers in brackets refer to the paper (`main.tex`). Units: seconds, metres, radians, newtons unless stated. Frames: `base` (robot base), `tool` (probe TCP; z = probe axis pointing into tissue), `image` (B-mode pixel grid).

---

## 1. Concepts and notation (paper Sec. III)

- Observation `o_t = (u_t, c_t, q_t)`: B-mode frame, probe-camera image (if present), robot state. Wrench `w_t` is logged and used by the controller and the when-head features.
- Action chunk `a_{t:t+H}`: future probe displacements over `H` seconds in the `tool` frame: tangential (x, y), orientation (rx, ry, rz). The probe axis z is owned by the force controller.
- Contact command `m̂_t ∈ {HOLD, SWEEP, LIFT, RECONTACT}`, output by the policy (or the when-head commit rule).
- Scheduler mode `m_t`: the protocol's command. Reaches the policy only in intent-aware arms (A5, DP-5) and baseline B6.
- Hold duration `D`: from hold start to the operator's key press (reaction time included). Cue time and key time are logged separately.
- Elapsed hold time `e`.

---

## 2. Theory module `usintent/theory/` (paper Eqs. 1–3, Fig. 1c)

Port `reference/rss_predictions.py` into tested functions. Each function takes a hold distribution object (`Uniform(a, b)`, `ShiftedExp(shift, mean)`, `Empirical(samples)`) with `sample(n, rng)`, `cdf(x)`, `median()`, `mean()`.

| Function | Definition |
|---|---|
| `hazard(dist, e, h)` | `P(D ≤ e+h \| D > e)` (Eq. 1) |
| `bound_median(dist)` | `E\|D − med(D)\|` (Eq. 2) |
| `err_calibrated_imitator(dist)` | `E\|D − D'\|`, with D, D' i.i.d. |
| `err_memoryless_hedger(dist)` | `E\|T − D\|`, T ~ Exp(mean = E[D]) independent of D; also `P(T < D)` |
| `pibar(dist, H)` | `E[min(H, D)] / E[D]` (Eq. 3) |
| `pibar_tau(dist, H, tau)` | `(H − tau)/E[D]` for `D ≥ H` (general form: `E[max(0, min(H, D) − tau)]/E[D]`) |

Golden values for tests, at 2M Monte-Carlo samples with tolerance ±0.02:
- Uniform[2,10]: bound 2.00, imitator 2.67, hedger 4.75 (early 0.60), π̄(H=1) 0.167, π̄_τ(0.2) 0.133, π̄_τ(0.4) 0.100, hazard(e=6, h=1) 0.25.
- 2+Exp(4): bound 2.77, imitator 4.00, hedger 5.16, hazard(e≥2, h=1) ≈ 0.221.

Also provide closed forms where they exist (uniform: `(b−a)/4`, `(b−a)/3`), and test the Monte-Carlo estimates against them.

---

## 3. Synthetic hold-world `usintent/sim/holdworld.py` (milestone M1)

A generator of synthetic episodes **in the exact dataset format** used for real data, with known ground truth. It is the test oracle for every downstream module.

- Episode: stations along a 1-D vessel path; sweep between stations at speed `v ∈ {v1, v2, v3}`; hold at each station with `D ~ p_D`; optional lift/recontact events; operator reaction time `R ~ LogNormal` added between cue and key.
- Kinematics: smooth velocity profiles (minimum-jerk accelerations of duration `t_acc`) so that a threshold-based departure detector has a real `tau`.
- Synthetic B-mode: low-resolution frames (e.g. 64×64) with a dark ellipse (vessel) on speckle-like noise. The frame content during a hold is **independent of D** (only i.i.d. speckle noise), except in an optional "image-adequacy leak" mode where a scalar `adequacy(t)` rises before the key press. This mode tests the when-head's `g_img` and the leakage classifier.
- Wrench: constant normal force + noise during contact.
- Logged fields: everything the real recorder logs (Sec. 6), including `cue_time`, `key_time`, `mode`, `station_index`, `episode_time`.
- Config: `p_D`, number of stations (random range), speeds, `t_acc`, chunk `H`, frame rate, noise levels, leak on/off, seed.

Acceptance: on 10k synthetic holds, the measured fraction of hold chunks containing a departure equals `pibar(p_D, H)` within ±0.01; the measured threshold-crossing rate matches `pibar_tau` with `tau` computed from `t_acc` and the threshold.

---

## 4. Data layer `usintent/data/`

- **Loaders** for LeRobot v3.0 datasets (Open-H robotic-ultrasound subsets and our own). Read the installed `lerobot` package for the current API; do not hard-code paths.
- **Chunk sampler**: yields `(obs history, action chunk)` with chunk starts sampled uniformly in time (the default in LeRobot-style training). Exposes per-chunk metadata: phase, `e`, whether the chunk contains a departure, exit type.
- **Phase segmentation**:
  - Cued data: phases come from the logged modes and key times.
  - Open-H data (no mode labels): `hold` = translational speed below `v_hold` for at least `T_hold_min`; `approach` = before first contact (from force if available, else from probe-axis motion); `contact_onset` = first `T_onset` seconds after contact; `sweep` = the rest during contact; transition windows of ±`w` s around each phase change. All thresholds come from `configs/data.yaml`, and `scripts/run_segmentation_sensitivity.py` reports results under ±50% threshold changes.
- **Splits**: by recording session and day. Assert that no session appears in two splits.
- **Ground-truth passthrough**: `data/passthrough.py` runs recorded actions through the exact decode/normalize/denormalize/frame-transform path used for policy outputs and asserts zero error (tolerance 1e-6).

---

## 5. Metrics `usintent/metrics/` (paper Sec. VI)

Open loop (per chunk, then aggregated per run):
- `chunk_error_trans`, `chunk_error_rot`: mean L2 over the chunk between predicted and recorded displacements.
- `ratio_to_cv`: policy error divided by B2 (constant-velocity) error **on the same chunks**, reported as a ratio of means per phase. Report the B2 error alongside; flag phases where B2 error is near zero (holds), since the ratio is unstable there.
- `hold_drift`: mean predicted displacement over chunks that start in a hold and contain no departure in ground truth.
- `departure_like_rate`: fraction of hold chunks whose predicted max displacement exceeds `thr` (from config: zero-motion baseline + measured controller jitter). Computed per exit type, excluding back-tracking holds for the direction test.
- `departure_direction`: cosine between predicted displacement and the ground-truth exit direction, per exit type.
- `rate_vs_e`: departure-like rate binned by elapsed hold time `e` (for A6, A7 and the hazard-shape test).

Closed loop (per trial; trial is the resampling unit):
- `timing_error = |T − D|` per hold; `T` = time of the first commanded motion above `thr` after hold start.
- Hold drift; fraction of time vessel in view; centring error (from a vessel segmentation, `metrics/vessel.py`, simple threshold + largest dark blob, validated against human labels on a subset); fraction of time force in band; mean confidence/adequacy in vessel ROI; count of safety-filter interventions.
- Trial success: all stations imaged with vessel in view ≥ `p_view` of each hold and force never leaves band.

---

## 6. Baselines `usintent/baselines/` (paper Table II)

Every baseline implements `predict(history) -> chunk` with the same interface as the policies.
- **B1 zero motion**: chunk of zeros.
- **B2 constant velocity / Kalman**: extrapolate the last estimated velocity (finite difference or a CV Kalman filter on pose) over H.
- **B3 IMM**: interacting multiple model filter with three modes: HOLD (zero velocity), CV (constant velocity, tangential), LIFT (velocity along probe axis), with a Markov transition matrix from config. Outputs mode probabilities and a chunk from the most probable mode. Report detection lag: time from key press until `P(new mode) > 0.5`.
- **B4 hazard-optimal blind**: knows `p_D` and `e`; predicts zero motion until `e ≥ med(D)`, then the mean departure chunk of the training data for the most frequent exit type. Attains Eq. 2 on cued data.
- **B5 speckle decorrelation**: out-of-plane motion from consecutive frames. Normalized cross-correlation `ρ` in patches, mapped to elevational displacement by a calibration curve `ρ(d) = exp(−d²/(2σ²))`. `σ` comes from config (calibrated on the rig later); on synthetic data use the generator's ground truth. Combine with in-plane motion from frame registration to form a chunk by constant-velocity extrapolation.
- **B6 stop-on-hold**: wraps any policy; whenever the scheduler mode is HOLD, it replaces the output with zeros and sets `m̂ = HOLD`.

---

## 7. Leakage classifier `usintent/leakage/` (paper H3a)

- Unit: time steps inside holds (subsampled), grouped by hold.
- Target: `ends_within(H)` (binary) and `remaining_time` (regression, optional).
- Feature sets:
  - `E`: elapsed time `e` only.
  - `E+ctx`: `e`, station index, number of stations, time since episode start.
  - `E+ctx+obs`: plus frozen image embeddings (a small pretrained CNN or the policy's frozen vision encoder; choose one, pin it), wrench statistics, image-adequacy features (Sec. 9).
- Models: logistic regression and gradient boosting.
- Evaluation: session-grouped cross-validation; log-loss and AUROC. The test statistic is the improvement of `E+ctx+obs` over `E`. A permutation test shuffles observation features within holds of matched `e` bins (1000 permutations).
- Run on (a) synthetic data with the leak off (expect no improvement), (b) synthetic data with the leak on (expect improvement), (c) Open-H natural data, (d) cued data.

---

## 8. Training-data intervention `usintent/intervene/` (paper H2 causal test)

- A weighted chunk sampler that sets the fraction of **hold chunks containing a departure** to a target `p ∈ {p_low, p_high}` (e.g. 0.5× and 2× the natural π̄) **without deleting data**. It reweights chunk-start sampling between "hold chunk with departure" and "hold chunk without departure"; all other chunks keep their natural weights.
- Reports the realized fraction over one epoch (must be within ±0.01 of target).
- Prediction: the departure-like rate of a trained policy scales with `p`. Analysis script: `scripts/run_intervention_analysis.py` fits rate ~ p across seeds and reports the slope with a permutation interval.

---

## 9. When-head `usintent/whenhead/` (paper Eq. 4, Fig. 2, arm A7)

- **Features**:
  - `kappa_t`: mean confidence-map value in the vessel ROI. Implement `confidence/karamalis.py` (random-walk confidence maps, Karamalis et al. 2012) on a downsampled frame, using a sparse solver. **Measure the runtime** and record it in `reports/`. If it is too slow for the policy rate, also implement `confidence/proxy.py` (depth-attenuation-compensated mean intensity under the transducer) and report the agreement between the two.
  - `s_t` coupling indicator: fraction of image columns whose near-field intensity exceeds a threshold (no-contact columns go dark). Threshold from config.
  - `f_t`: variance of normal force over the last 1 s.
  - `e`: elapsed hold time (from proprioception: time since speed fell below `v_hold`).
- **Model**: discrete-time hazard with bin width `Δ` (= policy period). `h_k = σ(g_img(κ, s, f) + g_int(e))`. `g_img` is a small MLP (2×32); `g_int` is a monotone-free spline or an MLP on `e`. Exit-type head: softmax over {SWEEP, LIFT}.
- **Loss**: survival negative log-likelihood per hold, `−Σ_k [y_k log h_k + (1 − y_k) log(1 − h_k)]`, with `y_k = 1` only in the bin containing the key press. Holds cut by episode end are right-censored (no event term).
- **Commit rule**: survival `S(e) = Π_j (1 − h_j)`; keep `m̂ = HOLD` while `1 − S(e) ≤ 0.5`; then switch to the predicted exit type. While HOLD is active, the action head is conditioned on the HOLD token and the output is forced to zero motion.
- **Image share**: `share_img = Var(g_img)/(Var(g_img) + Var(g_int))` over hold steps, plus the log-likelihood gain of the full model over `g_int` alone. Report it per dataset.
- Acceptance on synthetic data: leak off ⇒ share_img ≈ 0 and the commit time ≈ med(D) (timing error ≈ 2.00 s for Uniform[2,10]); leak on ⇒ share_img > 0 and timing error below 2.00 s.

---

## 10. Policies `usintent/policies/` (paper Sec. V)

- **GR00T-H** via Isaac-GR00T. Read the repo's fine-tuning entry point and data config (modality JSON) before writing anything. The VLM backbone is frozen; fine-tune the action head and embodiment projectors (confirm the flags in the repo). One config per arm in `configs/arms/`:
  - A1: B-mode + camera, single frame.
  - A2: A1 + B-mode history `k` frames (`k` from config).
  - A3: A1 + proprioception.
  - A5: A2 + mode embedding (inspect how conditioning enters the action head; add a learned embedding to it; document exactly where).
  - A6: A3 + time since last motion.
  - A7: A2 + when-head (Sec. 9), with proprioception for `e`.
  - Language-mode check: swap the instruction at test time and measure the change in predicted actions; report the arm as uninformative if the change is below the seed noise.
- **Diffusion Policy** via LeRobot: DP-1 (as A1) and DP-5 (as A5).
- Shared: same data, schedule and number of updates for every arm; ≥5 seeds for arms testing H2; seeds and git hash logged.
- Sampler controls at inference: stochastic vs deterministic initial noise; 4 vs 16 integration/denoising steps.
- Inference adapter: `predict(history, mode=None) -> (chunk, m̂)`. `mode` is accepted only by intent-aware arms and **asserted None** for intent-blind arms.

---

## 11. Statistics `usintent/stats/`

- Seed noise floor: the spread of A1 across seeds.
- Arm comparisons: every seed as a point; permutation test across runs (exact if small); effect relative to the noise floor.
- Closed loop: cluster bootstrap resampling whole trials; paired comparisons on the same station sequences and holds.
- Rank correlation between open-loop hold metrics and closed-loop timing across all runs and checkpoints.

---

## 12. ROS 2 packages `ros2_ws/src/`

- `us_intent_scheduler`: samples station count/order and `D` per hold from the configured distribution (per session block: uniform or shifted-exp); plays an audio cue at hold end; publishes `/protocol/mode_cue`; logs cue times. Deterministic given a seed (the seed is logged).
- `us_intent_teleop`: keyboard teleop over preset stations. A key press publishes `/protocol/mode_key` (mode + timestamp); the robot moves only after the key press.
- `us_intent_recorder`: rosbag2 (MCAP) recording profile for every topic in `docs/INTERFACES.md`, plus a converter from bag to LeRobot v3.0 that keeps cue/key/mode/station metadata, stores B-mode losslessly, and verifies timestamps are monotonic and on one clock.
- `us_intent_policy_server`: loads an arm checkpoint and runs inference at the policy rate. Enforces intent gating per arm (scheduler mode is wired only for A5/DP-5/B6). Publishes the chunk and `m̂`; logs every output.
- `us_intent_control` (Stage 2): hybrid force/position control in the `tool` frame (force along z with an impedance law, displacement tracking in x/y/rotation), a force set-point switched by `m̂`, and a CBF/limit filter on Cartesian velocity, acceleration, normal displacement and contact force, with intervention logging. Built on the UR ROS 2 driver; read its docs for the available controller interfaces (e.g. forward velocity / Cartesian controllers) and choose one. Developed and tested on mock hardware and URSim first. **All of `docs/SAFETY.md` applies.**

---

## 13. Paper outputs `scripts/`

- `make_fig_teaser_c.py`: predicted timing errors (theory) for Fig. 1c, checked against `reference/rss_predictions.py`.
- `make_table_phase_ratio.py`: open-loop error ratio to B2 by phase (prelim check 1).
- `make_fig_hedging.py`: predicted π̄_τ vs measured departure-like rate, including intervention arms (prelim check 2 and H2).
- `make_fig_rate_vs_e.py`: departure rate vs e under both hold distributions.
- `make_table_leakage.py`: leakage results on synthetic, Open-H and cued data.
- `make_table_closed_loop.py`: timing error per arm vs predictions; image/force metrics; interventions.
- Every output embeds the git hash and the config hash.
