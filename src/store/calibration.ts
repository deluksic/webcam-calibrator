// Calibration reactive store — Solid.js 2.0
import { createStore } from 'solid-js';

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
  cameraReady: boolean;
  cameraWidth: number;
  cameraHeight: number;
  detectedCornerCount: number;
  detectionOverlay: Array<{ x: number; y: number }>;
  views: ObservedView[];
  minViewsNeeded: number;
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
  setCalibrationStore(s => {
    s.views = [];
    s.intrinsics = null;
    s.distortion = null;
    s.rmsError = null;
    s.status = 'idle';
    s.errorMessage = null;
    s.detectedCornerCount = 0;
    s.detectionOverlay = [];
  });
}

export function addView(view: ObservedView) {
  setCalibrationStore(s => { s.views.push(view); });
}

export function setCameraReady(ready: boolean, width?: number, height?: number) {
  setCalibrationStore(s => {
    s.cameraReady = ready;
    if (width !== undefined) s.cameraWidth = width;
    if (height !== undefined) s.cameraHeight = height;
  });
}

export function setDetectionOverlay(corners: Array<{ x: number; y: number }>) {
  setCalibrationStore(s => {
    s.detectionOverlay = corners;
    s.detectedCornerCount = corners.length;
  });
}

export function setCalibrationResult(
  intrinsics: CameraIntrinsics,
  distortion: DistortionCoeffs,
  rmsError: number
) {
  setCalibrationStore(s => {
    s.intrinsics = intrinsics;
    s.distortion = distortion;
    s.rmsError = rmsError;
    s.status = 'done';
  });
}

export function setStatus(status: CalibrationState['status'], errorMessage?: string) {
  setCalibrationStore(s => {
    s.status = status;
    s.errorMessage = errorMessage ?? null;
  });
}
