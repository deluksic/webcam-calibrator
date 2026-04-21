import { createSignal } from 'solid-js'

import { CalibrationView } from '@/components/CalibrationView'
import { CameraStreamProvider } from '@/components/camera/CameraStreamContext'
import { DebugView } from '@/components/DebugView'
import { ResultsView } from '@/components/ResultsView'
import { TargetView } from '@/components/TargetView'
import { VERSION } from '@/version'

import styles from '@/components/App.module.css'

export type View = 'target' | 'calibrate' | 'results' | 'debug'

export function App() {
  const [view, setView] = createSignal<View>('calibrate')

  return (
    <CameraStreamProvider>
      <div class={styles.root}>
        <nav class={styles.nav}>
          <button
            class={[styles.navBtn, { [styles.navBtnActive]: view() === 'target' }]}
            onClick={() => setView('target')}
          >
            Target
          </button>
          <button
            class={[styles.navBtn, { [styles.navBtnActive]: view() === 'calibrate' }]}
            onClick={() => setView('calibrate')}
          >
            Calibrate
          </button>
          <button
            class={[styles.navBtn, { [styles.navBtnActive]: view() === 'results' }]}
            onClick={() => setView('results')}
          >
            Results
          </button>
          <button
            class={[styles.navBtn, { [styles.navBtnActive]: view() === 'debug' }]}
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
  )
}
