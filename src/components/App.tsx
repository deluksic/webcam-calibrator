import { createSignal } from 'solid-js';
import CalibrationView from './CalibrationView';
import TargetView from './TargetView';
import ResultsView from './ResultsView';
import styles from './App.module.css';

export type View = 'target' | 'calibrate' | 'results';

export default function App() {
  const [view, setView] = createSignal<View>('calibrate');

  return (
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
      </nav>

      <main class={styles.main}>
        {view() === 'target'    && <TargetView />}
        {view() === 'calibrate' && <CalibrationView />}
        {view() === 'results'   && <ResultsView />}
      </main>
    </div>
  );
}
