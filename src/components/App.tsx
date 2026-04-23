import { A, HashRouter, Route, type RouteSectionProps } from '@solidjs/router'

import { CalibrationView } from '@/components/CalibrationView'
import { CameraStreamProvider } from '@/components/camera/CameraStreamContext'
import { DebugView } from '@/components/DebugView'
import { ResultsView } from '@/components/ResultsView'
import { TargetView } from '@/components/TargetView'
import { VERSION } from '@/version'

import styles from '@/components/App.module.css'

export function App() {
  const Layout = (props: RouteSectionProps) => (
    <>
      <nav class={styles.nav}>
        <A href="/target" class={styles.navBtn}>Target</A>
        <A href="/calibrate" class={styles.navBtn}>Calibrate</A>
        <A href="/results" class={styles.navBtn}>Results</A>
        <A href="/debug" class={styles.navBtn}>Debug</A>
        <span class={styles.version}>{VERSION}</span>
      </nav>
      <main class={styles.main}>{props.children}</main>
    </>
  )

  return (
    <CameraStreamProvider>
      <HashRouter root={Layout}>
        <Route path="/" component={TargetView} />
        <Route path="/target" component={TargetView} />
        <Route path="/calibrate" component={CalibrationView} />
        <Route path="/results" component={ResultsView} />
        <Route path="/debug" component={DebugView} />
      </HashRouter>
    </CameraStreamProvider>
  )
}