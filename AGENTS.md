# realsense-yolov8-nitros-bridge — agent notes

RealSense → Isaac ROS NITROS/YOLOv8 perception front end. **Reference docs live
in `README.md`** — it is a copy-boundary analysis (where each memcpy happens,
what NITROS zero-copy does and doesn't cover, the IPC compatibility matrix),
plus a `## Notes` section per source file. Read it before changing the pipeline;
the current shape is the conclusion of that analysis, not an accident.

**The ROS package name is `realsense_yolov8_nitros_bridge`, not the directory
name** — `--packages-select realsense-yolov8-nitros-bridge` selects nothing.

**Shadowed by `/workspaces/ros2_ws`** (`Dockerfile.thornbots` LAYER 5 copies
this directory in and builds it alongside our six other packages; editing it
rebuilds all seven, so iterate with `colcon build` in the container instead).
Once built locally, a `src/` edit is live under `dexec.sh` but not in the
user's terminal. Confirm with
`../isaac_ros_common/scripts/dexec.sh -- ros2 pkg prefix realsense_yolov8_nitros_bridge`.
C++, so a source change always needs a rebuild; `--symlink-install` won't help.

Launch-file/config edits interact with IPC: the launch file is what enables
intra-process comms, and getting it wrong costs a full frame copy rather than
producing an error. Verify against `README.md` §2 before changing composition.

## Scope

- Owns the camera → NITROS → YOLOv8 path and its `/detections_output`.
  Per-detection depth/bearing belongs to `../Realsense_ROI_Depth_Rectifier`;
  target selection and tracking to `../thornbots_pkg`.

## Committing

This package is a submodule of `thornbots_workspace`, on branch `main`. Commit
and push here first, then bump this gitlink in `../` — one logical change, one
bump, never a gitlink pointing at an unpushed commit. Full rule in
`../CLAUDE.md` § Packages.
