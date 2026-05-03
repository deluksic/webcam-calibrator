import { For, createMemo } from 'solid-js'

import { cameraDeviceScore } from '@/components/camera/cameraDeviceScore'
import { useCameraStream } from '@/components/camera/CameraStreamContext'
import { RESOLUTION_LADDER, resolutionLabel, type Resolution } from '@/components/camera/cameraStreamAcquire'

import styles from '@/components/camera/CameraStreamSelects.module.css'

export function CameraStreamSelects() {
  const cam = useCameraStream()
  const devicesSorted = createMemo(() => {
    const list = cam.devices()
    return [...list].sort((a, b) => cameraDeviceScore(b) - cameraDeviceScore(a))
  })

  return (
    <div class={styles.root}>
      <select
        class={[styles.select, styles.deviceSelect]}
        value={cam.selectedCameraDeviceId() ?? ''}
        onChange={(e) => cam.setSelectedCameraDeviceId(e.currentTarget.value)}
      >
        <For each={devicesSorted()}>
          {(item) => (
            <option value={item().deviceId}>{item().label || `Camera ${item().deviceId.slice(0, 8)}`}</option>
          )}
        </For>
      </select>
      <select
        class={styles.select}
        value={cam.selectedResolution()}
        onChange={(e) => cam.setSelectedResolution(e.currentTarget.value as Resolution)}
      >
        <For each={Object.keys(RESOLUTION_LADDER)} keyed={false}>
          {(resolution) => <option value={resolution()}>{resolutionLabel(resolution())}</option>}
        </For>
      </select>
    </div>
  )
}
