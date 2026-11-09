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

## Remaining
- See bottom of this file after the eval run.
