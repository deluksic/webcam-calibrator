import { createSignal } from 'solid-js';
import { CalibrationView } from './CalibrationView';
import { DebugView } from './DebugView';
import { TargetView } from './TargetView';
import { ResultsView } from './ResultsView';
import { CameraStreamProvider } from './camera/CameraStreamContext';
import styles from './App.module.css';
import { VERSION } from '../version';

export type View = 'target' | 'calibrate' | 'results' | 'debug';

export function App() {
  const [view, setView] = createSignal<View>('calibrate');

  return (
    <CameraStreamProvider>
      <div class={styles.root}>
        <nav class={styles.nav}>
          <button
            class={[styles.navBtn, view() === 'target' && styles.navBtnActive]}
            onClick={() => setView('target')}
          >
            Target
          </button>
          <button
            class={[styles.navBtn, view() === 'calibrate' && styles.navBtnActive]}
            onClick={() => setView('calibrate')}
          >
            Calibrate
          </button>
          <button
            class={[styles.navBtn, view() === 'results' && styles.navBtnActive]}
            onClick={() => setView('results')}
          >
            Results
          </button>
          <button
            class={[styles.navBtn, view() === 'debug' && styles.navBtnActive]}
            onClick={() => setView('debug')}
          >
            Debug
          </button>
          <span class={styles.version}>{VERSION}</span>
        </nav>

        <main class={styles.main}>
          {view() === 'target' && <TargetView />}
          {view() === 'calibrate' && <CalibrationView />}
          {view() === 'results' && <ResultsView />}
          {view() === 'debug' && <DebugView />}
        </main>
      </div>
    </CameraStreamProvider>
  );
}
