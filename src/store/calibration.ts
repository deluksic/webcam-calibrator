// Calibration reactive store — Solid.js 2.0
import { createStore } from 'solid-js/store';

// ─── Types ───────────────────────────────────────
export interface CameraIntrinsics {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface DistortionCoeffs {
  k1: number;
  k2: number;
  k3: number;
  p1: number;
  p2: number;
}

export interface ObservedView {
  imagePoints: number[];  // flattened [u0,v0, u1,v1, ...] in pixels
  worldPoints: number[];  // flattened [X0,Y0, Z0, X1,Y1,Z1, ...] in meters
  reprojectionError: number;
}

export interface CalibrationState {
  // Camera stream
  cameraReady: boolean;
  cameraWidth: number;
  cameraHeight: number;

  // Detection
  detectedCornerCount: number;
  detectionOverlay: Array<{ x: number; y: number }>;

  // Views collected
  views: ObservedView[];
  minViewsNeeded: number;

  // Calibration result
  intrinsics: CameraIntrinsics | null;
  distortion: DistortionCoeffs | null;
  rmsError: number | null;
  status: 'idle' | 'collecting' | 'solving' | 'done' | 'error';
  errorMessage: string | null;
}

// ─── Store ───────────────────────────────────────
const [calibrationStore, setCalibrationStore] = createStore<CalibrationState>({
  cameraReady: false,
  cameraWidth: 1280,
  cameraHeight: 720,
  detectedCornerCount: 0,
  detectionOverlay: [],
  views: [],
  minViewsNeeded: 10,
  intrinsics: null,
  distortion: null,
  rmsError: null,
  status: 'idle',
  errorMessage: null,
});

export { calibrationStore, setCalibrationStore };

// ─── Actions ─────────────────────────────────────
export function resetCalibration() {
  setCalibrationStore({
    views: [],
    intrinsics: null,
    distortion: null,
    rmsError: null,
    status: 'idle',
    errorMessage: null,
    detectedCornerCount: 0,
    detectionOverlay: [],
  });
}

export function addView(view: ObservedView) {
  setCalibrationStore('views', (v) => [...v, view]);
}

export function setCameraReady(ready: boolean, width?: number, height?: number) {
  setCalibrationStore('cameraReady', ready);
  if (width !== undefined) setCalibrationStore('cameraWidth', width);
  if (height !== undefined) setCalibrationStore('cameraHeight', height);
}

export function setDetectionOverlay(corners: Array<{ x: number; y: number }>) {
  setCalibrationStore('detectionOverlay', corners);
  setCalibrationStore('detectedCornerCount', corners.length);
}

export function setCalibrationResult(
  intrinsics: CameraIntrinsics,
  distortion: DistortionCoeffs,
  rmsError: number
) {
  setCalibrationStore({ intrinsics, distortion, rmsError, status: 'done' });
}

export function setStatus(status: CalibrationState['status'], errorMessage?: string) {
  setCalibrationStore({ status, errorMessage: errorMessage ?? null });
}