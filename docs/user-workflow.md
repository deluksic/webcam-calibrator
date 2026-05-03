# User workflow (short)

Use **two roles**: a **display** (shows the AprilTag target) and a **camera** (captures it). They can be different devices or the same machine.

Copy in the app follows two bands: **progress** (keep going—you are not being told anything “failed” just because you need more views), and **needs attention** (hard solver problems such as bad geometry—then the UI states that clearly).

## 1. Display — Target

1. Open the app and go to **Target**.
2. Use **Target** on a **separate screen** or **print** it so the board is full-size for the camera.
3. Avoid glare and motion blur. Changing grid size or **Randomize Tags** invalidates a calibration session on the camera side — finish or **Reset** on **Calibrate** before changing IDs.
4. Optional tips on Home/Target can be dismissed; use **Show intro again** / reopen controls if you hid them.

## 2. Camera — Calibrate

1. Open **Calibrate** on the camera device (phone, laptop, etc.).
2. Allow camera access, pick device and resolution.
3. The dashed **focus guide** on the feed is advisory framing only—it does **not** gate **Snapshot**. **Start** stays disabled until **at least two decoded tag IDs** appear in the frame (**numbered** IDs, not **?**).
4. Press **Start** once those IDs are stable. The session stays in memory if you switch tabs (**Home**, **Target**, **Results**); only **Reset** clears it. There is **no Pause/Stop**—you can leave the tab while capture is **running**.
5. Move the camera for different viewpoints. Press **Snapshot** to add frames to the pool (the first qualifying frame can also be added automatically right after **Start**). The **guidance** block under the controls updates with progress; failed snapshot attempts show there until state changes (no timed dismiss).
6. Some detections may show **?** when the pattern looks plausible but the tag ID is not read yet; only **numbered** IDs count toward layout and snapshots.
7. The solver runs in the background. When calibration looks stable and at least **four** solver-ready frames qualify, **Results** unlocks—open it to orbit the board, export JSON, or save to the library from there.

## 3. Results

1. **Saved calibrations** lists saved runs and a built-in **Demo calibration** row so the 3D view is never empty on first load. Pick **Show** on a row to drive the orbit view, or **Use latest solve** to show the latest calibration from **Calibrate** when available.
2. **Drag** (or pinch / scroll) to orbit. **Export JSON** exports the calibration currently shown (selected library entry, or **latest** when no library row is selected).
3. While **latest** is active (not viewing a saved row): **Save to library** stores the current calibration (same readiness rules as unlocking **Results**), and **Continue Calibration** links back to **Calibrate**. Those actions are hidden when a saved entry is selected—use **Use latest solve** first if needed.
4. **Compare** on saved rows adds them to the comparison set (letters **A**, **B**, …). As soon as at least one **ok** run is in the set, a small intrinsics table appears (one column first—add more runs for side-by-side columns). **Clear comparison** empties the set.

## 4. Reset

**Reset** on **Calibrate** clears the capture pool, layout, and the shared **latest** result used by **Results** when you rely on “latest.” It does **not** delete **saved library** entries (remove those from **Results**). It does not change **Target** settings. **Reset** is disabled when there is nothing to clear yet.

Starting a new capture session with **Start** clears any library row selection so **Results** tracks **latest** again.

## Nav indicator

While a session has work in progress (**running** or a non-empty frame pool), the **Calibrate** nav item shows a small marker and an accessible name so you notice an active session after switching pages.
