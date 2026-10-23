# Webcam Calibrator

In-browser camera calibration against a **hex target** of black and white markers. Marker detection, identity matching, and the live pipeline run on the **WebGPU** stack in your browser; there is no server.

[Webcam Calibrator](https://deluksic.github.io/webcam-calibrator)

**Requires:** a recent Chromium-class browser with WebGPU (Chrome/Edge 113+, or Firefox Nightly with WebGPU enabled).

## What you can do

- **Home** — Quick orientation and entry points.
- **Target** — Show the fixed full-screen hex target on a display device or second screen.
- **Calibrate** — Point the camera at the target, collect frames, and run calibration. Identified markers are overlaid live; **Reset** clears the current run and shared “latest” result.
- **Results** — After a good solve, explore the refined target points in a simple 3D view and **export** a self-describing hex calibration JSON (intrinsics, distortion, poses, target markers, per-frame observations, and frame images).
- **Point Lab / ROI Lab** — Diagnostics for the hex pattern, blob detector, and identity recovery.

The target is editable on the Target page (columns, rows, points per cell edge, marker scale, color seed, and resolution). The sidebar renders a QR code for a Calibrate link containing the full target configuration, so opening that link on the camera device reproduces the exact same target.

For how it works under the hood, see [`docs/architecture.md`](docs/architecture.md) and [`docs/plan.md`](docs/plan.md). The identity ICP recovery algorithm is documented in [`docs/point-identity-icp-recovery.md`](docs/point-identity-icp-recovery.md).

**Dev:** Solid.js, Vite — `pnpm dev`, `pnpm build`, `pnpm test`.
