# Suggested edits to the Stage-1 extended abstract (main_clean.pdf)

Text suggestions only; the paper is not in this repo. Ordered by importance.

1. **Intent signal timing (Sec. III/IV, Fig. 1c).** State that intent-aware arms (A5, B6) receive
   the operator's key event, so their timing error is the system latency. If you keep the cue
   instead, Fig. 1c's A5 bar must become "latency + reaction-time spread".
2. **Realised p_D (Sec. III, Prediction 1).** The scheduler draws the cue offset; D (start → key
   press) = drawn duration + reaction time. Say that the Stage-2 predictions (Fig. 1c, π̄, π̄_τ)
   are computed from the realised D, which check (4) measures.
3. **Hedger assumption (Prediction 1).** Add "assuming the sampled departure starts at a uniformly
   random point of the chunk" to "whatever H and the replanning interval".
4. **Lift-off and departure-like (Sec. IV/VI).** The action has no probe-axis component, so the
   pilot's "large motions along the probe axis" cannot occur by construction. A spurious lift
   shows as m̂ = LIFT, and departure-like chunks are counted as tangential displacement > thr **or**
   m̂ ∈ {LIFT, RECONTACT}.
5. **τ with three speeds (Prediction 2).** τ depends on sweep speed; report π̄_τ with the mean τ
   over speed levels (exact, linear in τ) or per speed.
6. **H3(a) wording.** The code now has both versions (remaining-time regression and "ends within
   H"); keep "remaining hold time" and mention the binary variant, or vice versa.
7. **Closed-loop D (Sec. VI).** Say the operator still hears the cue and presses the key in
   closed-loop trials, so D means the same thing as in training.
8. **Controller during collection (Sec. IV, VIII).** State which controller runs during data
   collection. If it is the M9 controller, its validation must precede collection, not Stage-2
   weeks 1–2.
9. **H3(b) phrasing.** 20 trials per arm (~100 holds) separate 4.75 / 2.67 / 2.00 s but cannot show
   A7 ≈ B4. Phrase it as an ordering with confidence intervals.
10. **GR00T details (Sec. V).** Fine-tuning used LeRobot-v2 export, state dropout 0 and the mode as
    a learned state embedding entering the action head's DiT tokens. One clause; reviewers ask
    "where does the mode enter".
11. **Intervention (Sec. III, H2).** Implemented for the diffusion policy. If GR00T is meant to carry
    it, its episode sampler needs a change; otherwise say "on the diffusion policy".
12. **Platform placeholders.** Telemed model, phantom models, and the F/T source (separate 6-axis
    sensor vs the UR7e's flange sensing).
13. **Second phantom (optional, Sec. VIII).** One sentence: natural, uncued holds on a central-line
    neck phantom, where the hold end can be visible (needle in vein), measure the when-head's
    image share.
