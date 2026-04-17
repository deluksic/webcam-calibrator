# webcam-calibrator

A pure WebGPU-powered webcam calibration tool running entirely in the browser. Detects AprilTag36h11 calibration targets, extracts quad corners via homography, and collects observations for lens distortion calibration.

**Stack:** Solid.js, TypeGPU (WebGPU), Vite

**Requires:** Chrome/Edge 113+ or Firefox Nightly with WebGPU enabled (`chrome://flags/#enable-unsafe-webgpu`).

## Views

- **Target** — generates a printable AprilTag36h11 grid (6×6 encoding, 587 tag IDs) as an SVG
- **Calibrate** — live camera feed with GPU pipeline and display modes (Gray, Edges, NMS, Labels, Grid, Debug)
- **Results** — calibration output (intrinsics, distortion params, export)

## Status

- Target generation: complete
- Live detection + corner extraction: complete
- Calibration math (PnP + bundle adjustment): not yet implemented
