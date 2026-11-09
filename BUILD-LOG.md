# BUILD-LOG

One time-boxed Claude Code session, 2026-11-09, start 14:19 EST. Simulation only.

## Done (clock times EST)
- 14:19–14:22 repo + `git init`, uv env (mujoco 4.0, numpy, opencv-python-headless, pytest, torch CPU wheel), Panda copied from `cell-culture-workcell-twin/assets` (Menagerie, Apache-2.0).
- 14:22–14:25 `scene/panda_suction.xml` (Menagerie `panda_nohand.xml` + vacuum tool: shaft + pad geom + TCP site), `build_scene.py` -> `scene/cell.xml` (conveyor pick zone, pallet, 12 free case bodies of 4 types, one inactive `weld` per case, 2 fixed RGB-D cameras + overview camera), `sim.py`.
  - Found: weld `relpose` convention (case pose in the tool frame) by checking `efc_pos` = 0 at activation.
  - Holding check first read the weld's `efc_force`; it also carried the contact squeeze of the pad pressing the case, so every pick "dropped". Replaced with the standard sizing formula F = m(a - g) and peel torque |r x F| per step (case acceleration from smoothed velocity differences).
- 14:25–14:27 `planner.py` (rule-aware greedy + first-fit baseline + checker), `vision.py`, `metrics.py`, `state_machine.py`.
- 14:27–14:41 debugging the cell until a pallet builds cleanly:
  - Joint-space descents swung sideways and knocked neighbours -> straight-line TCP moves (`move_linear`, IK at 10 points).
  - Moving pallet -> conveyor from a low pose swept through stacks -> always lift to SAFE_Z first.
  - IK landed in odd branches / failed at the far pallet corner -> elbow-up seed with joint 7 set for the wanted tool yaw; pallet moved 3-4 cm closer to the robot; IK failure in TRANSFER now fails the case (logged) instead of moving low.
  - Arm occluded the cameras -> pick camera moved to the side (oblique view), arm parks above the pick spot before VERIFY.
  - Pad collision re-enabled while overlapping a just-released case flipped it -> pad collision stays off until the next approach.
- 14:41 grasp sweep started (3 workers), `eval.py`, `tests/`, `make_results.py`, `render_pallet.py` written while it ran.

## Cuts (vs IDEA.md / BUILD.md)
- No ROS 2, no PCL, no cases/hour claim, one arm, 50 pallets (as BUILD.md says).
- Gap metric is an axis-aligned estimate from detected centres and footprints, not a full contour distance.

## Honest limits
- Vacuum physics is a model: holding force = vacuum pressure x sealed pad area (overlap of the pad disc with the case top), peel lever = 0.4 x pad radius. Values are made up but plausible; drops and grasp labels come from this model.
- Grasp offsets are injected (pad positioning error, sigma 12 mm) and treated as measured by the pick camera before suction; the grasp model sees that measured offset. The hidden per-pick vacuum level (leak) is the noise it cannot see.
- Arm motion uses MuJoCo physics with the Panda's position actuators (so payload sag shows up as placement error), but there is no collision-aware planner: moves are straight lines at a safe height.
- The planner is a greedy heuristic; it can break a rule through its soft fallback (counted, see RESULTS.md).
- The first-fit baseline is only checked by the rule checker, not executed in physics.
- Cases are fed from a sequencing buffer in the order the planner asks for (a teleport to the pick spot stands in for the conveyor).

## Done (continued)
- 14:45 sweep done (360 picks: 220 held, 131 peel, 9 pull); MLP trained (held-out 0.956 vs rule 0.700).
- 14:45–14:58 eval, 50 pallets, 3 workers (745 s wall; machine load average 17–32 from the parallel build).
- 14:58–15:01 pytest 9/9, render, `failure_breakdown.py`, `RESULTS.md`, README, commit. End 15:01 (~42 min).

## Result in one line
22 / 50 pallets fully correct. Failed pallets: drops (peel) 13 events, 32 unstable cases (knocked or
shifted > 20 mm / tilted > 5 deg after settle), 16 cases the pallet camera could not find (usually the
same knocked cases), 1 IK failure. Planner 0 rule violations vs first-fit 149 on the same cases.
The "stable pallets" claim holds for the rule side (planner) but the physical success rate is 44 %.

## Remaining
- Find why cases get knocked (likely the carried case or pad brushing neighbours on descent with
  2-3 cm gaps, and joint-space moves between the safe-height waypoints): add a collision check on
  the carried case, larger cell pitch, or slower final descent; re-run the eval.
- Compare drops with vs without grasp-model gating (`eval.py --no-model --tag _nomodel`) — not run (time).
- Vision mis-detections of knocked cases inflate p95 placement error; split vision error vs true error.

## Round 2 (15:00–15:16, machine free)
- Motion fix (v2): lift vertically to max(0.50, tallest stack top + carried case height + 0.10 m), one smooth
  joint move over the target at that height, vertical descent to just above the place pose, release,
  vertical retreat. IK on straight-line moves now seeds from the previous waypoint (stays on one branch).
- Tried first: a level straight-line (20-waypoint) transfer. Piecewise-linear joint commands gave
  acceleration spikes and many pull/peel drops (37 in the first ~8 pallets), so it was killed and replaced
  by the single smooth joint move above. Not in the results.
- Old run kept as `results/*_v1.*`. Re-ran the same 50 pallets (seed 0) with the grasp model on
  (`eval_summary.json`) and off (`eval_summary_nomodel.json`, no pre-pick check at all; the offset rule was not run in the eval).
- Before -> after (model on): 22 -> 36 / 50 pallets; unstable cases 32 -> 0; not found by camera 16 -> 0;
  placement error mean 9.11 -> 4.76 mm, median 4.84 -> 4.62 mm; peel drops 13 -> 13.
- Model on vs off (v2): 36 vs 23 / 50 pallets; peel drops 13 vs 36.
- Camera "not found" cases disappeared with the motion fix (they were knocked cases), so no camera change was needed.
- Note: the grasp model was trained on the v1 sweep (lift + joint move); not retrained for v2.
- Remaining: the 13 peel drops (lower transfer acceleration for heavy cases, or a seal check that
  re-measures the offset after contact); the one IK-unreachable target; a run with the offset rule as the gate.
