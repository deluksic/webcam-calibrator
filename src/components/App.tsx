import { A, HashRouter, Route, type RouteSectionProps } from '@solidjs/router'

import { Home } from '@/components/Home'
import { CalibrationView } from '@/components/CalibrationView'
import { CameraStreamProvider } from '@/components/camera/CameraStreamContext'
import { DebugView } from '@/components/DebugView'
import { ResultsView } from '@/components/ResultsView'
import { TargetView } from '@/components/TargetView'
import { VERSION } from '@/version'

import styles from '@/components/App.module.css'

import { useLocation } from '@solidjs/router'

export function App() {
  const location = useLocation()

  const isHome = location.pathname === '/' || location.pathname === '/target'

  const Layout = (props: RouteSectionProps) => (
    <>
      <nav class={styles.nav}>
        <A href="/" class={isHome ? styles.navBtnActive : styles.navBtn}>
          Home
        </A>
        <A href="/target" class={isHome ? styles.navBtn : styles.navBtn}>
          Target
        </A>
        <A href="/calibrate" class={isHome ? styles.navBtn : styles.navBtn}>
          Calibrate
        </A>
        <A href="/results" class={isHome ? styles.navBtn : styles.navBtn}>
          Results
        </A>
        <span class={styles.version}>{VERSION}</span>
      </nav>
      <main class={styles.main}>{props.children}</main>
    </>
  )

  return (
    <CameraStreamProvider>
      <HashRouter root={Layout}>
        <Route path="/" component={Home} />
        <Route path="/target" component={TargetView} />
        <Route path="/calibrate" component={CalibrationView} />
        <Route path="/results" component={ResultsView} />
        <Route path="/debug" component={DebugView} />
      </HashRouter>
    </CameraStreamProvider>
  )
}
