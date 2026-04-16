import { createSignal, createMemo } from 'solid-js';
import { generateTagGridSVG } from '../lib/april-tag-gen';
import styles from './TargetView.module.css';

export default function TargetView() {
  const [cols, setCols] = createSignal(4);
  const [rows, setRows] = createSignal(3);
  const [tagSize, setTagSize] = createSignal(40);
  const [spacing, setSpacing] = createSignal(1.5);
  const [checkerboard, setCheckerboard] = createSignal(true);
  const [isFullscreen, setIsFullscreen] = createSignal(false);

  const svg = createMemo(() =>
    generateTagGridSVG({
      cols: cols(),
      rows: rows(),
      tagSize: tagSize(),
      spacing: spacing(),
      checkerboard: checkerboard(),
      margin: 1,
    })
  );

  const handleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  // Listen for fullscreen change
  if (typeof document !== 'undefined') {
    document.addEventListener('fullscreenchange', () => {
      setIsFullscreen(!!document.fullscreenElement);
    });
  }

  return (
    <div class={styles.root}>
      <div class={styles.display}>
        <div
          class={styles.svgContainer}
          innerHTML={svg()}
          onClick={handleFullscreen}
          title="Click to toggle fullscreen"
        />
        <button class={styles.fullscreenBtn} onClick={handleFullscreen}>
          {isFullscreen() ? 'Exit Fullscreen' : 'Fullscreen'}
        </button>
      </div>

      <aside class={styles.controls}>
        <h2 class={styles.title}>Target Settings</h2>

        <div class={styles.field}>
          <label class={styles.label}>Grid Size</label>
          <div class={styles.row}>
            <span class={styles.fieldLabel}>Columns</span>
            <select
              class={styles.select}
              value={String(cols())}
              onChange={(e) => setCols(parseInt(e.currentTarget.value))}
            >
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
              <option value="6">6</option>
            </select>
            <span class={styles.fieldLabel}>Rows</span>
            <select
              class={styles.select}
              value={String(rows())}
              onChange={(e) => setRows(parseInt(e.currentTarget.value))}
            >
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
            </select>
          </div>
        </div>

        <div class={styles.field}>
          <label class={styles.label}>Tag Size (mm)</label>
          <input
            type="number"
            class={styles.input}
            value={String(tagSize())}
            min={10}
            max={200}
            onChange={(e) => setTagSize(parseInt(e.currentTarget.value) || 40)}
          />
        </div>

        <div class={styles.field}>
          <label class={styles.label}>Spacing</label>
          <select
            class={styles.select}
            value={String(spacing())}
            onChange={(e) => setSpacing(parseFloat(e.currentTarget.value))}
          >
            <option value="1">1× tag size</option>
            <option value="1.25">1.25× tag size</option>
            <option value="1.5">1.5× tag size</option>
            <option value="2">2× tag size</option>
          </select>
        </div>

        <div class={styles.field}>
          <label class={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={checkerboard()}
              onChange={(e) => setCheckerboard(e.currentTarget.checked)}
            />
            Include checkerboard between tags
          </label>
        </div>

        <div class={styles.info}>
          <p>Click the target to toggle fullscreen mode.</p>
          <p>Print at 100% scale for accurate sizing.</p>
        </div>
      </aside>
    </div>
  );
}