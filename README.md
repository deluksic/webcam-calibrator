# webcam-calibrator

A WebGPU-based browser app for working with AprilTag36h11 calibration targets: live detection, quad fitting, homography, dictionary decode, and a pool of per-tag observations for a lens model.

**Stack:** Solid.js, TypeGPU (WebGPU), Vite

**Requires:** Chrome/Edge 113+ or Firefox Nightly with WebGPU enabled (`chrome://flags/#enable-unsafe-webgpu`).

## Views

- **Target** — printable AprilTag36h11 grid (6×6 payload, 587 IDs) as SVG
- **Calibrate** — **grid** mode only: live feed with homography overlay and tag decode; **Start / Pause / Reset**; top‑K observation pool and run statistics (edge histogram is on **Debug**, not here)
- **Debug** — all pipeline display modes (Gray, Edges, NMS, Labels, Grid, **Debug**), edge histogram, fallback quads option, log tail
- **Results** — screen for camera intrinsics, distortion, and export when a solver is connected (not wired in this build)

Details: [`ARCHITECTURE.md`](ARCHITECTURE.md), [`docs/plan.md`](docs/plan.md).
