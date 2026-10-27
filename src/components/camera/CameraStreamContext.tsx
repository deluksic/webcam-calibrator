import type { Setter, Accessor, ParentProps } from 'solid-js'
import { createContext, createMemo, createSignal, latest, onSettled, untrack, useContext } from 'solid-js'
import { createEffect } from 'solid-js'

import { cameraDeviceScore } from './cameraDeviceScore'
import type { CameraResolution } from './cameraStreamAcquire'
import {
  DEFAULT_RESOLUTION,
  acquireVideoStream,
  listVideoInputDevices,
  primeCameraPermission,
  resolutionFromKey,
  resolutionKey,
  stopCameraStream,
  supportedResolutions,
} from './cameraStreamAcquire'

const { navigator } = globalThis
const QUERY = new URLSearchParams(globalThis.location.search)
const INITIAL_RESOLUTION = resolutionFromKey(QUERY.get('resolution')) ?? DEFAULT_RESOLUTION

export type CameraStreamContextValue = {
  /** Sorted videoinput devices (async memo — `await devices()` in `<Show>`). */
  devices: Accessor<MediaDeviceInfo[]>
  selectedCameraDeviceId: Accessor<string | undefined>
  setSelectedCameraDeviceId: (id: string) => void
  stream: Accessor<MediaStream | undefined>
  /** Canonical `WxH` key of the requested capture resolution. */
  selectedResolution: Accessor<string>
  setSelectedResolution: Setter<string>
  /** Candidate modes the selected camera reports it can reach. */
  resolutionOptions: Accessor<readonly CameraResolution[]>
  registerCameraUsage: (id: symbol) => void
  unregisterCameraUsage: (id: symbol) => void
}

export const CameraStreamContext = createContext<CameraStreamContextValue>()

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
  const [registeredCameraUsers, setRegisteredCameraUsers] = createSignal(new Set<symbol>(), { equals: false })
  const cameraIsNeeded = createMemo(() => registeredCameraUsers().size > 0)
  const cameraIsNeededLatch = createMemo<boolean>((prev) => prev || cameraIsNeeded())
  const [selectedResolution, setSelectedResolution] = createSignal<string>(resolutionKey(INITIAL_RESOLUTION))

  const [devices, setDevices] = createSignal<MediaDeviceInfo[]>(async () => {
    if (!cameraIsNeededLatch()) {
      return []
    }
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

  const deviceIdAndStream = createMemo<{ deviceId: string; resolution: string; stream: MediaStream } | undefined>(
    async (prev) => {
      const id = selectedCameraDeviceId()
      const resolution = selectedResolution()
      if (!id || !cameraIsNeeded()) {
        return undefined
      }
      if (prev?.deviceId === id && prev.resolution === resolution) {
        return prev
      }
      // Many mobile browsers only allow one active camera: release before a new getUserMedia.
      if (prev?.stream) {
        stopCameraStream(prev.stream)
      }
      const stream = await acquireVideoStream(id, resolutionFromKey(resolution) ?? DEFAULT_RESOLUTION)
      return { deviceId: id, resolution, stream }
    },
  )

  createEffect(deviceIdAndStream, (pair) => {
    if (!pair) {
      return
    }
    const { deviceId, resolution, stream } = pair
    return () => {
      untrack(() => {
        if (
          stream &&
          (!latest(cameraIsNeeded) ||
            latest(selectedCameraDeviceId) !== deviceId ||
            latest(selectedResolution) !== resolution)
        ) {
          stopCameraStream(stream)
        }
      })
    }
  })

  const stream = () => deviceIdAndStream()?.stream
  const supportedCameraResolutions = createMemo(() => supportedResolutions(stream()))

  const resolutionOptions = createMemo(() => {
    const options = supportedCameraResolutions()
    const selected = resolutionFromKey(selectedResolution())
    if (!selected) {
      return options
    }
    const selectedKey = resolutionKey(selected)
    if (options.some((option) => resolutionKey(option) === selectedKey)) {
      return options
    }
    // Keep a URL-requested mode selectable even when it is outside the
    // camera's reported range; getUserMedia will pick the closest real mode.
    return [...options, selected].sort((a, b) => a.width * a.height - b.width * b.height)
  })

  // A shared link (or a quick click before the first stream arrives) may ask
  // for a mode this camera cannot reach. Snap to the largest supported mode
  // that fits, so the select never keeps advertising an impossible size.
  createEffect(
    () => ({
      supported: supportedCameraResolutions(),
      selected: resolutionFromKey(selectedResolution()),
    }),
    ({ supported, selected }) => {
      if (!selected || supported.length === 0) {
        return
      }
      const selectedKey = resolutionKey(selected)
      if (supported.some((option) => resolutionKey(option) === selectedKey)) {
        return
      }
      const fitting = supported.filter((option) => option.width <= selected.width && option.height <= selected.height)
      const snapped = fitting.at(-1) ?? supported[0]!
      setSelectedResolution(resolutionKey(snapped))
    },
  )

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
    selectedResolution,
    setSelectedResolution,
    resolutionOptions,
    registerCameraUsage,
    unregisterCameraUsage,
  }

  return <CameraStreamContext value={value}>{props.children}</CameraStreamContext>
}
