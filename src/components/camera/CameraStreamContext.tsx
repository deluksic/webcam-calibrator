import {
  type Accessor,
  type ParentProps,
  createContext,
  createMemo,
  createSignal,
  onCleanup,
  useContext,
} from 'solid-js'

import { acquireVideoStream, listVideoInputDevices, primeCameraPermission } from '@/lib/cameraStreamAcquire'

function deviceScore(d: MediaDeviceInfo): number {
  const label = d.label.toLowerCase()
  let score = 0
  if (label.includes('back') || label.includes('rear')) score += 100
  if (label.includes('wide')) score += 50
  if (label.includes('ultra')) score += 30
  if (label.includes('tele')) score -= 20
  if (label.includes('front') || label.includes('user')) score -= 100
  return score
}

export type CameraStreamContextValue = {
  /** Sorted videoinput devices (async memo — `await devices()` in `<Show>`). */
  devices: Accessor<Promise<MediaDeviceInfo[]>>
  deviceId: Accessor<string | undefined>
  setDeviceId: (id: string) => void
  stream: Accessor<Promise<MediaStream | undefined>>
  streamError: Accessor<string | undefined>
  /** Last known intrinsic size from `getSettings()` after open/upgrade. */
  trackSize: Accessor<{ w: number; h: number }>
}

const CameraStreamContext = createContext<CameraStreamContextValue>()

export function useCameraStream(): CameraStreamContextValue {
  const v = useContext(CameraStreamContext)
  if (!v) {
    throw new Error('useCameraStream must be used within CameraStreamProvider')
  }
  return v
}

export function CameraStreamProvider(props: ParentProps) {
  const [deviceId, setDeviceId] = createSignal<string | undefined>()
  const [streamError, setStreamError] = createSignal<string | undefined>(undefined, {
    ownedWrite: true,
  })
  const [trackSize, setTrackSize] = createSignal({ w: 1280, h: 720 }, { ownedWrite: true })
  const [deviceListEpoch, setDeviceListEpoch] = createSignal(0)

  const devices = createMemo(async () => {
    deviceListEpoch()
    await primeCameraPermission()
    const list = await listVideoInputDevices()
    return [...list].sort((a, b) => deviceScore(b) - deviceScore(a))
  })

  const stream = createMemo(async () => {
    let acquired: MediaStream | undefined
    onCleanup(() => {
      acquired?.getTracks().forEach((t) => t.stop())
    })

    setStreamError(undefined)
    const list = await devices()
    const id = deviceId() ?? list[0]?.deviceId
    if (!id) return undefined

    try {
      const s = await acquireVideoStream(id)
      acquired = s
      const track = s.getVideoTracks()[0]
      const st = track?.getSettings()
      if (st?.width && st?.height) {
        setTrackSize({ w: st.width, h: st.height })
      }
      return s
    } catch (e) {
      setStreamError(e instanceof Error ? e.message : String(e))
      return undefined
    }
  })

  const onDeviceChange = () => {
    setDeviceListEpoch((n) => n + 1)
  }

  if (typeof navigator !== 'undefined' && navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', onDeviceChange)
    onCleanup(() => navigator.mediaDevices.removeEventListener('devicechange', onDeviceChange))
  }

  const value: CameraStreamContextValue = {
    devices: devices as unknown as Accessor<Promise<MediaDeviceInfo[]>>,
    deviceId,
    setDeviceId,
    stream: stream as unknown as Accessor<Promise<MediaStream | undefined>>,
    streamError,
    trackSize,
  }

  return <CameraStreamContext value={value}>{props.children}</CameraStreamContext>
}
