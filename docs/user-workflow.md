# User workflow (short)

Use **two roles**: a **display** (shows the fixed hex target) and a **camera** (captures it). They can be different devices or the same machine.

Copy in the app follows two bands: **progress** (keep going—you are not being told anything “failed” just because you need more views), and **needs attention** (hard solver problems such as bad geometry—then the UI states that clearly).

## 1. Display — Target

1. Open the app and go to **Target**.
2. Use **Fullscreen target** on a **separate screen** (or the same machine) so the marker field is large in the camera view.
3. Configure columns, rows, points per cell edge, marker scale, color seed, and resolution. The sidebar QR code opens Calibrate with the complete configuration embedded in the URL; scan it on the camera device.
4. Avoid glare and motion blur. Optional tips on Home can be dismissed; use **Show intro again** if you hid them.

## 2. Camera — Calibrate

1. Open **Calibrate** on the camera device (phone, laptop, etc.).
2. Allow camera access, pick device and resolution.
3. The focus guide is advisory framing only—it does **not** gate **Snapshot**. **Start** stays disabled until **at least 12 markers** are identified and assigned stable design IDs in the frame.
4. Press **Start** once those markers are stable. The session stays in memory if you switch tabs; only **Reset** clears it. There is **no Pause/Stop**—you can leave the tab while capture is **running**.
5. Move the camera for different viewpoints. Press **Snapshot** to add frames to the pool (the first qualifying frame can also be added automatically right after **Start**). The guidance block updates with progress; failed snapshot attempts show there until state changes.
6. Markers that cannot be identified are drawn as unresolved rings and are not used as calibration measurements. Identity conflicts are reported in the guidance panel.
7. The solver runs in the background. When calibration looks stable and at least **four** solver-ready frames qualify, **Results** unlocks—open it to orbit the refined target points, export JSON, or save to the library.

## 3. Results

1. **Saved calibrations** lists saved runs and a built-in **Demo calibration** row so the 3D view is never empty on first load. Pick **Show** on a row to drive the orbit view, or **Use latest solve** to show the latest calibration from **Calibrate** when available.
2. **Drag** (or pinch / scroll) to orbit. **Export JSON** exports the calibration currently shown (selected library entry, or **latest** when no library row is selected).
3. While **latest** is active: **Save to library** stores the current calibration, and **Continue Calibration** links back to **Calibrate**. Those actions are hidden when a saved entry is selected—use **Use latest solve** first if needed.
4. **Compare** on saved rows adds them to the comparison set (letters **A**, **B**, …). **Clear comparison** empties the set.
5. Exported hex JSON is schema v2: it includes the target configuration, refined markers, K/distortion, per-frame poses, point observations, and frame PNGs. Existing AprilTag JSON exports remain valid files but use the older schema.

## 4. Reset

**Reset** on **Calibrate** clears the capture pool and the shared **latest** result used by **Results** when you rely on “latest.” It does **not** delete **saved library** entries (remove those from **Results**). It does not change the target. **Reset** is disabled when there is nothing to clear yet.

Starting a new capture session with **Start** clears any library row selection so **Results** tracks **latest** again.

## Nav indicator

While a session has work in progress (**running** or a non-empty frame pool), the **Calibrate** nav item shows a small marker and an accessible name so you notice an active session after switching pages.
