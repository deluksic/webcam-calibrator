import type { Component } from 'solid-js'
import { For, Show } from 'solid-js'

import type { DetectedQuad } from '@/gpu/detectedQuad'
import type { CustomTagOverlaySession } from '@/lib/customTagOverlaySession'
import { displayLabelForTagId } from '@/lib/tag36h11'

import styles from '@/components/camera/LiveCameraPipeline.module.css'

const { max, abs } = Math

export type { CustomTagOverlaySession }

export const TagIdGridOverlay: Component<{
  quads: DetectedQuad[]
  scale: { x: number; y: number }
  customTagOverlay?: () => CustomTagOverlaySession
}> = (props) => {
  return (
    <For each={props.quads} keyed={false}>
      {(quad) => {
        const c = () => quad().corners
        const cx = () => (c()[0].x + c()[1].x + c()[2].x + c()[3].x) / 4
        const cy = () => (c()[0].y + c()[1].y + c()[2].y + c()[3].y) / 4
        const height = () =>
          max(abs(c()[0].y - c()[1].y), abs(c()[1].y - c()[2].y), abs(c()[2].y - c()[3].y), abs(c()[3].y - c()[0].y))

        const label = () => {
          const id = quad().decodedTagId
          if (id === undefined) {
            return undefined
          }
          if (id < 0) {
            const ot = props.customTagOverlay?.()
            if (!ot) {
              return '*'
            }
            if (!ot.collectionRunning) {
              return '*'
            }
            const idx = ot.sessionIndexByCustomTagId.get(id)
            return idx === undefined ? '*' : `*${idx}`
          }
          return displayLabelForTagId(id)
        }
        const id = () => quad().decodedTagId
        const customStyled = () => typeof id() === 'number' && id()! < 0 && props.customTagOverlay !== undefined
        return (
          <Show when={label() !== undefined}>
            <div
              class={[styles.tagIdOverlay, customStyled() && styles.tagIdOverlayCustom]}
              style={{
                '--tag-x': `${cx() * props.scale.x}px`,
                '--tag-y': `${cy() * props.scale.y}px`,
                '--tag-size': `${height() * props.scale.y}px`,
              }}
            >
              {label()}
            </div>
          </Show>
        )
      }}
    </For>
  )
}
