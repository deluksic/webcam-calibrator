# Webcam calibrator — product notes

## Overview

In-browser camera calibration against a fixed hex marker target; no server. Capture, ONNX segmentation, marker identity, and calibration all run client-side.

**Stack:** SolidJS 2.0, TypeGPU (WebGPU), ONNX Runtime Web, CSS modules.

---

## UI

| View          | Role                                                                                                                                 |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Home**      | Two-role landing copy; links to Target / Calibrate; dismissible tips                                                                 |
| **Target**    | Editable hex target (columns, rows, points, seed, scale, resolution); QR link to Calibrate; JSON export                         |
| **Calibrate** | Live ROI/identity feed; **Start** only with ≥12 identified markers; snapshot/auto-first; frame pool; worker solve + live reprojection; guidance + collapsed advanced metrics |
| **Results**   | WebGPU orbit view of refined target points; **Export JSON** (hex schema v2); **Save to library** & **Continue Calibration**            |
| **Point Lab** | Hex pattern, fingerprint, and local homography/ICP experiments                                                                        |
| **ROI Lab**   | ONNX mask, blob segmentation, component, label, and identity diagnostics with a seed scan                                              |

**Camera** — shared [`CameraStreamContext`](../src/components/camera/CameraStreamContext.tsx) at the root. **Calibrate** capture: [`HexCameraPipeline`](../src/components/camera/HexCameraPipeline.tsx). Removed routes `/hex-target`, `/debug`, `/gradient-profiles` are not redirected.

---

## Detection pipeline

1. ROI ONNX mask at 256 px wide with aspect-following height.
2. Masked full-resolution luma histogram → black / gray / white cuts.
3. 2×2 color-aware opening, pointer-jump connected components, centroid NMS.
4. Observed proximity graph + binary-color fingerprints.
5. Hash identity match; local homography/ICP recovery for ambiguous markers.
6. Accepted point observations feed the calibration frame pool.

Details: [`architecture.md`](architecture.md), [`hex-blob-detection.md`](hex-blob-detection.md), [`point-identity-icp-recovery.md`](point-identity-icp-recovery.md).

---

## Calibration

- The Calibrate page reads the target config from URL parameters produced by the Target page QR link.
- Object points are normalized by the median design spacing (`z = 0`); units are scale-free.
- [`calibrationInput.ts`](../src/lib/calibrationInput.ts) requires ≥3 views, ≥8 points per view, and ≥8 shared design IDs.
- [`calibration.worker.ts`](../src/workers/calibration.worker.ts) uses OpenCV `calibrateCameraRO` and returns K, rational distortion, per-frame poses, refined target points, and RMS.
- Results unlock at ≥4 valid solve frames.

---

## Results export

New exports are hex **schema v2** with `target` parameters, marker colors and refined marker positions, K/distortion, per-frame poses, point observations, and frame PNGs. Existing AprilTag JSON exports remain valid for the existing differential renderer; a future hex differential renderer can consume schema v2. See [`architecture.md`](architecture.md#results-and-export).

---

## Roadmap

- Build the hex differential renderer.
- Persist the calibration library to `localStorage`.
- Inline numeric intrinsics/distortion tables and richer compare views.
