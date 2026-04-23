import type { Setter, Accessor, ParentProps } from 'solid-js'
import { createContext, createMemo, createSignal, latest, onSettled, untrack, useContext } from 'solid-js'
import { createEffect } from 'solid-js'

import { cameraDeviceScore } from './cameraDeviceScore'
import type { Resolution } from './cameraStreamAcquire'
import {
  acquireVideoStream,
  listVideoInputDevices,
  primeCameraPermission,
  RESOLUTION_LADDER,
  stopCameraStream,
} from './cameraStreamAcquire'

const { navigator } = globalThis

export type CameraStreamContextValue = {
  /** Sorted videoinput devices (async memo — `await devices()` in `<Show>`). */
  devices: Accessor<MediaDeviceInfo[]>
  selectedCameraDeviceId: Accessor<string | undefined>
  setSelectedCameraDeviceId: (id: string) => void
  stream: Accessor<MediaStream | undefined>
  trackSize: Accessor<{ width: number; height: number } | undefined>
  selectedResolution: Accessor<Resolution>
  setSelectedResolution: Setter<Resolution>
  registerCameraUsage: (id: symbol) => void
  unregisterCameraUsage: (id: symbol) => void
}

const CameraStreamContext = createContext<CameraStreamContextValue>()

export function useCameraStream(): CameraStreamContextValue {
  const id = Symbol('camera-usage')
  const v = useContext(CameraStreamContext)
  if (!v) {
    throw new Error('useCameraStream must be used within CameraStreamProvider')
  }
  onSettled(() => {
    v.registerCameraUsage(id)
    return () => {
      v.unregisterCameraUsage(id)
    }
  })
  return v
}

export function CameraStreamProvider(props: ParentProps) {
  console.log('render')
  const [registeredCameraUsers, setRegisteredCameraUsers] = createSignal(new Set<symbol>(), { equals: false })
  const cameraIsNeeded = createMemo(() => registeredCameraUsers().size > 0)
  const [selectedResolution, setSelectedResolution] = createSignal<Resolution>('low')

  const [devices, setDevices] = createSignal<MediaDeviceInfo[]>(async () => {
    await primeCameraPermission()
    const list = await listVideoInputDevices()
    return [...list].sort((a, b) => cameraDeviceScore(b) - cameraDeviceScore(a))
  })

  const [selectedCameraDeviceId, setSelectedCameraDeviceId] = createSignal<string | undefined>((prev) => {
    const devices_ = devices()
    if (devices_.find((d) => d.deviceId === prev)) {
      return prev
    }
    return devices_[0]?.deviceId
  })

  const deviceIdAndStream = createMemo<{ deviceId: string; stream: MediaStream } | undefined>(async (prev) => {
    const id = selectedCameraDeviceId()
    console.log('deviceID', id, cameraIsNeeded(), prev)
    if (!id || !cameraIsNeeded()) {
      return undefined
    }
    if (prev?.deviceId === id) {
      return prev
    }
    const stream = await acquireVideoStream(id, untrack(selectedResolution))
    console.log('stream acquired', id, cameraIsNeeded(), prev)
    return { deviceId: id, stream }
  })

  createEffect(deviceIdAndStream, (pair) => {
    if (!pair) {
      return
    }
    const { deviceId, stream } = pair
    return () => {
      untrack(() => {
        if (stream && (!latest(cameraIsNeeded) || latest(selectedCameraDeviceId) !== deviceId)) {
          console.log('stopped stream', deviceId)
          stopCameraStream(stream)
        }
      })
    }
  })

  const stream = () => deviceIdAndStream()?.stream

  const trackSize = createMemo(async () => {
    const s = stream()
    const settings = s?.getVideoTracks()[0]?.getSettings()
    const { width, height } = settings ?? {}
    if (s === undefined || width === undefined || height === undefined) {
      console.log('no stream')
      return undefined
    }
    const res = RESOLUTION_LADDER[selectedResolution()]
    if (width !== res.width.ideal || height !== res.height.ideal) {
      await s?.getVideoTracks()[0]?.applyConstraints({ width: res.width, height: res.height })
      const settings = s.getVideoTracks()[0]?.getSettings()
      const { width, height } = settings ?? {}
      if (width === undefined || height === undefined) {
        return undefined
      }
      console.log('renegotiated', width, height)
      return { width, height }
    }
    console.log('got', width, height)
    return { width, height }
  })

  async function onDeviceChange() {
    setDevices(await navigator.mediaDevices.enumerateDevices())
  }

  onSettled(() => {
    navigator.mediaDevices.addEventListener('devicechange', onDeviceChange)
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', onDeviceChange)
    }
  })

  function registerCameraUsage(id: symbol) {
    setRegisteredCameraUsers((p) => {
      p.add(id)
      return p
    })
  }

  function unregisterCameraUsage(id: symbol) {
    setRegisteredCameraUsers((p) => {
      p.delete(id)
      return p
    })
  }

  const value: CameraStreamContextValue = {
    devices,
    selectedCameraDeviceId,
    setSelectedCameraDeviceId,
    stream,
    trackSize,
    selectedResolution,
    setSelectedResolution,
    registerCameraUsage,
    unregisterCameraUsage,
  }

  return <CameraStreamContext value={value}>{props.children}</CameraStreamContext>
}
