import {
  For,
  Show,
  createMemo,
  createSignal,
  isPending,
} from "solid-js";
import {
  LiveCameraPipeline,
  type DisplayMode,
} from "./camera/LiveCameraPipeline";
import { useCameraStream } from "./camera/CameraStreamContext";
import styles from "./DebugView.module.css";
import pipelineStyles from "./camera/LiveCameraPipeline.module.css";

function deviceScore(d: MediaDeviceInfo): number {
  const label = d.label.toLowerCase();
  let score = 0;
  if (label.includes("back") || label.includes("rear")) score += 100;
  if (label.includes("wide")) score += 50;
  if (label.includes("ultra")) score += 30;
  if (label.includes("tele")) score -= 20;
  if (label.includes("front") || label.includes("user")) score -= 100;
  return score;
}

export function DebugView() {
  const cam = useCameraStream();
  const [logs, setLogs] = createSignal<string[]>([]);
  const [displayMode, setDisplayMode] = createSignal<DisplayMode>("grid");
  const [showFallbacks, setShowFallbacks] = createSignal(false);
  const showGrid = () => true;

  const log = (msg: string) => {
    Promise.resolve().then(() => {
      setLogs((prev) => [
        ...prev.slice(-8),
        `${new Date().toISOString().slice(11, 19)} ${msg}`,
      ]);
      console.log(msg);
    });
  };

  const devicesSorted = createMemo(async () => {
    const list = await cam.devices();
    return [...list].sort((a, b) => deviceScore(b) - deviceScore(a));
  });

  const toolbar = () => (
    <div class={styles.debugToolbar}>
      <Show when={() => !isPending(() => devicesSorted())}>
        <select
          class={[pipelineStyles.cameraSelect, styles.debugCameraSelect]}
          value={cam.deviceId() ?? ""}
          onChange={(e) => cam.setDeviceId(e.currentTarget.value)}
        >
          <Show when={devicesSorted()}>
            {(d) => (
              <For each={d()}>
                {(item) => (
                  <option value={item().deviceId}>
                    {item().label || `Camera ${item().deviceId.slice(0, 8)}`}
                  </option>
                )}
              </For>
            )}
          </Show>
        </select>
      </Show>
      <div class={styles.debugModeRow}>
        <button
          type="button"
          class={
            displayMode() === "grayscale"
              ? pipelineStyles.modeButtonActive
              : pipelineStyles.modeButton
          }
          onClick={() => setDisplayMode("grayscale")}
        >
          Gray
        </button>
        <button
          type="button"
          class={
            displayMode() === "edges"
              ? pipelineStyles.modeButtonActive
              : pipelineStyles.modeButton
          }
          onClick={() => setDisplayMode("edges")}
        >
          Edges
        </button>
        <button
          type="button"
          class={
            displayMode() === "nms"
              ? pipelineStyles.modeButtonActive
              : pipelineStyles.modeButton
          }
          onClick={() => setDisplayMode("nms")}
        >
          NMS
        </button>
        <button
          type="button"
          class={
            displayMode() === "labels"
              ? pipelineStyles.modeButtonActive
              : pipelineStyles.modeButton
          }
          onClick={() => setDisplayMode("labels")}
        >
          Labels
        </button>
        <button
          type="button"
          class={
            displayMode() === "grid"
              ? pipelineStyles.modeButtonActive
              : pipelineStyles.modeButton
          }
          onClick={() => setDisplayMode("grid")}
        >
          Grid
        </button>
        <label class={pipelineStyles.checkboxLabel}>
          <input
            type="checkbox"
            checked={showFallbacks()}
            onChange={(e) => setShowFallbacks(e.currentTarget.checked)}
          />
          Fallbk
        </label>
        <button
          type="button"
          class={
            displayMode() === "debug"
              ? pipelineStyles.modeButtonActive
              : pipelineStyles.modeButton
          }
          onClick={() => setDisplayMode("debug")}
        >
          Debug
        </button>
      </div>
    </div>
  );

  return (
    <div class={styles.root}>
      {cam.streamError() ? (
        <p style={{ color: "var(--color-error)" }}>
          Camera: {cam.streamError()}
        </p>
      ) : null}
      <LiveCameraPipeline
        displayMode={displayMode}
        showGrid={showGrid}
        showFallbacks={showFallbacks}
        showHistogramCanvas={() => true}
        stream={cam.stream}
        trackSize={cam.trackSize}
        onLog={log}
        toolbar={toolbar}
      />
      <div class={styles.logTail}>
        {logs().map((line) => (
          <div>{line}</div>
        ))}
      </div>
    </div>
  );
}
