import { A, HashRouter, Route, type RouteSectionProps, useCurrentMatches } from '@solidjs/router'
import { createMemo } from 'solid-js'

import { CalibrationLatestProvider } from '@/components/calibration/CalibrationLatestContext'
import { CalibrationView } from '@/components/CalibrationView'
import { CameraStreamProvider } from '@/components/camera/CameraStreamContext'
import { DebugView } from '@/components/DebugView'
import { Home } from '@/components/Home'
import { ResultsView } from '@/components/results/ResultsView'
import { TargetView } from '@/components/TargetView'
import { VERSION } from '@/version'

import styles from '@/components/App.module.css'

export function App() {
  const Layout = (props: RouteSectionProps) => {
    const matches = useCurrentMatches()
    const currentPath = createMemo(() => matches()[0]?.path || '/')

    const isHome = createMemo(() => currentPath() === '/')
    const isTarget = createMemo(() => currentPath() === '/target')
    const isCalibrate = createMemo(() => currentPath() === '/calibrate')
    const isResults = createMemo(() => currentPath() === '/results')
    const isDebug = createMemo(() => currentPath() === '/debug')

    return (
      <>
        <nav class={styles.nav}>
          <A href="/" class={[styles.navBtn, isHome() && styles.navBtnActive]}>
            Home
          </A>
          <A href="/target" class={[styles.navBtn, isTarget() && styles.navBtnActive]}>
            Target
          </A>
          <A href="/calibrate" class={[styles.navBtn, isCalibrate() && styles.navBtnActive]}>
            Calibrate
          </A>
          <A href="/results" class={[styles.navBtn, isResults() && styles.navBtnActive]}>
            Results
          </A>
          <A href="/debug" class={[styles.navBtn, isDebug() && styles.navBtnActive]}>
            Debug
          </A>
          <span class={styles.version}>{VERSION}</span>
        </nav>
        <main class={styles.main}>{props.children}</main>
      </>
    )
  }

  return (
    <CameraStreamProvider>
      <CalibrationLatestProvider>
        <HashRouter root={Layout}>
          <Route path="/" component={Home} />
          <Route path="/target" component={TargetView} />
          <Route path="/calibrate" component={CalibrationView} />
          <Route path="/results" component={ResultsView} />
          <Route path="/debug" component={DebugView} />
        </HashRouter>
      </CalibrationLatestProvider>
    </CameraStreamProvider>
  )
}
