import type { Component } from 'solid-js'
import { For, createMemo } from 'solid-js'

import type { DetectedQuad } from '@/gpu/contour'
import type { CustomTagOverlaySession } from '@/lib/customTagOverlaySession'
import { displayLabelForTagId } from '@/lib/tag36h11'

import styles from '@/components/camera/LiveCameraPipeline.module.css'

const { max, abs } = Math

export type { CustomTagOverlaySession }

export type Bbox = {
  minX: number
  minY: number
  maxX: number
  maxY: number
  area: number
}

export function QuadCandidateOverlay(props: { bboxes: Bbox[]; scale: { x: number; y: number } }) {
  const candidates = createMemo(() => {
    const MIN_AREA = 40 * 40
    const MAX_AREA = 200000
    const MIN_AR = 0.3
    const MAX_AR = 3.5
    return props.bboxes.filter((b) => {
      const w = b.maxX - b.minX
      const h = b.maxY - b.minY
      if (w <= 0 || h <= 0) {
        return false
      }
      const area = w * h
      if (area < MIN_AREA || area > MAX_AREA) {
        return false
      }
      const ar = w / h
      if (ar < MIN_AR || ar > MAX_AR) {
        return false
      }
      return true
    })
  })

  return (
    <For each={candidates()} keyed={false}>
      {(box) => (
        <div
          class={styles.bbox}
          style={{
            '--bbox-x': `${box().minX * props.scale.x}px`,
            '--bbox-y': `${box().minY * props.scale.y}px`,
            '--bbox-w': `${(box().maxX - box().minX) * props.scale.x}px`,
            '--bbox-h': `${(box().maxY - box().minY) * props.scale.y}px`,
          }}
        />
      )}
    </For>
  )
}

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
          const q = quad()
          const id = q.decodedTagId
          if (typeof id !== 'number') {
            // Dict miss after filter = fully binary unknown codeword — not red “?”.
            return '*'
          }
          const ot = props.customTagOverlay?.()
          if (id < 0 && ot) {
            if (!ot.collectionRunning || !ot.firstCustomTakeDone) {
              return '*'
            }
            const idx = ot.sessionIndexByCustomTagId.get(id)
            return idx === undefined ? '?' : `*${idx}`
          }
          return displayLabelForTagId(id)
        }
        const id = () => quad().decodedTagId
        const customStyled = () => typeof id() === 'number' && id()! < 0 && props.customTagOverlay !== undefined
        return (
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
        )
      }}
    </For>
  )
}
