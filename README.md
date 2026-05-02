# webcam-calibrator

In-browser camera calibration against **AprilTag36h11** targets. Detection and the live pipeline run on the **WebGPU** stack in your browser; there is no server.

**Requires:** a recent Chromium-class browser with WebGPU (Chrome/Edge 113+, or Firefox Nightly with WebGPU enabled).

## What you can do

- **Home** — Quick orientation and entry points.
- **Target** — Generate a printable or full-screen tag sheet.
- **Calibrate** — Point the camera at the target, collect frames, and run calibration. You get a live overlay on the video and session stats; **Reset** clears the current run and shared “latest” result.
- **Results** — After a good solve, explore the board in a simple 3D view and **export** camera parameters (intrinsics, distortion, poses) as JSON.
- **Debug** — Extra display modes, histogram, and logs for tuning detection.

For how it works under the hood, see [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`docs/plan.md`](docs/plan.md).

**Dev:** Solid.js, Vite — `pnpm dev`, `pnpm build`, `pnpm test`.
