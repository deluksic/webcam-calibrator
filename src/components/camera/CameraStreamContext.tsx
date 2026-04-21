import {
  type Accessor,
  type ParentProps,
  createContext,
  createMemo,
  createSignal,
  onCleanup,
  useContext,
} from 'solid-js'

import { cameraDeviceScore } from './cameraDeviceScore'
import { acquireVideoStream, listVideoInputDevices, primeCameraPermission } from './cameraStreamAcquire'

const { navigator } = globalThis

export type CameraStreamContextValue = {
  /** Sorted videoinput devices (async memo — `await devices()` in `<Show>`). */
  devices: Accessor<MediaDeviceInfo[]>
  deviceId: Accessor<string | undefined>
  setDeviceId: (id: string) => void
  stream: Accessor<MediaStream | undefined>
  trackSize: Accessor<{ width: number; height: number } | undefined>
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
  const [selectedCameraDeviceId, setSelectedCameraDeviceId] = createSignal<string>()
  const [deviceListEpoch, setDeviceListEpoch] = createSignal(0)

  const devices = createMemo(async () => {
    deviceListEpoch()
    await primeCameraPermission()
    const list = await listVideoInputDevices()
    return [...list].sort((a, b) => cameraDeviceScore(b) - cameraDeviceScore(a))
  })

  const stream = createMemo(async () => {
    const list = devices()
    const id = selectedCameraDeviceId() ?? list[0]?.deviceId
    if (!id) {
      return undefined
    }
    onCleanup(() => {
      stream?.getTracks().forEach((t) => t.stop())
    })
    const stream = await acquireVideoStream(id)
    return stream
  })

  const trackSize = createMemo(() => {
    const settings = stream()?.getVideoTracks()[0]?.getSettings()
    const { width, height } = settings ?? {}
    if (width === undefined || height === undefined) {
      return undefined
    }
    return { width, height }
  })

  const onDeviceChange = () => {
    setDeviceListEpoch((n) => n + 1)
  }

  if (typeof navigator !== 'undefined' && navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', onDeviceChange)
    onCleanup(() => navigator.mediaDevices.removeEventListener('devicechange', onDeviceChange))
  }

  const value: CameraStreamContextValue = {
    devices,
    deviceId: selectedCameraDeviceId,
    setDeviceId: setSelectedCameraDeviceId,
    stream,
    trackSize,
  }

  return <CameraStreamContext value={value}>{props.children}</CameraStreamContext>
}
