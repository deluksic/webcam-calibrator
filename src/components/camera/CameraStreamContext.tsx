import type { Setter , Accessor, ParentProps} from 'solid-js'
import { createContext, createMemo, createSignal, onCleanup, useContext } from 'solid-js'

import { cameraDeviceScore } from './cameraDeviceScore'
import type { Resolution } from './cameraStreamAcquire'
import {
  acquireVideoStream,
  listVideoInputDevices,
  primeCameraPermission,
  RESOLUTION_LADDER,
} from './cameraStreamAcquire'

const { navigator } = globalThis

export type CameraStreamContextValue = {
  /** Sorted videoinput devices (async memo — `await devices()` in `<Show>`). */
  devices: Accessor<MediaDeviceInfo[]>
  deviceId: Accessor<string | undefined>
  setDeviceId: (id: string) => void
  stream: Accessor<MediaStream | undefined>
  trackSize: Accessor<{ width: number; height: number } | undefined>
  setSelectedResolution: Setter<Resolution | undefined>
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
  const [selectedResolution, setSelectedResolution] = createSignal<Resolution>()
  const [deviceListEpoch, setDeviceListEpoch] = createSignal(0)

  const devices = createMemo(async () => {
    deviceListEpoch()
    await primeCameraPermission()
    const list = await listVideoInputDevices()
    return [...list].sort((a, b) => cameraDeviceScore(b) - cameraDeviceScore(a))
  })

  const stream = createMemo<MediaStream | undefined>(
    async (prev) => {
      const list = devices()
      const id = selectedCameraDeviceId() ?? list[0]?.deviceId
      if (!id) {
        return undefined
      }

      const resolution = selectedResolution()
      if (prev && resolution) {
        const res = RESOLUTION_LADDER[resolution]
        await prev.getVideoTracks()[0]?.applyConstraints({ width: res.width, height: res.height })
        return prev
      }

      const stream = await acquireVideoStream(id, resolution)
      return stream
    },
    { equals: false },
  )

  onCleanup(() => {
    stream()
      ?.getTracks()
      ?.forEach((t) => t.stop())
  })

  const trackSize = createMemo(() => {
    const settings = stream()?.getVideoTracks()[0]?.getSettings()
    const { width, height } = settings ?? {}
    if (width === undefined || height === undefined) {
      return undefined
    }
    console.log(width, height)
    return { width, height }
  })

  const onDeviceChange = () => {
    setDeviceListEpoch((n) => n + 1)
  }

  if (typeof navigator !== 'undefined' && navigator.mediaDevices?.addEventListener) {
    navigator.mediaDevices.addEventListener('devicechange', onDeviceChange)
    onCleanup(() => navigator.mediaDevices.removeEventListener('devicechange', onDeviceChange))
  }

  const value = {
    devices,
    deviceId: selectedCameraDeviceId,
    setDeviceId: setSelectedCameraDeviceId,
    stream,
    trackSize,
    setSelectedResolution,
  }

  return <CameraStreamContext value={value}>{props.children}</CameraStreamContext>
}
