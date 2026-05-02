import { For, createSignal, createMemo } from 'solid-js'

import { selectRandomTags } from '@/lib/april-tag-gen'
import { gridVizFillRgbCss } from '@/lib/hashStableColor'
import { TAG36H11_COUNT, tagIdPattern } from '@/lib/tag36h11'

import styles from '@/components/TargetView.module.css'

export function TargetView() {
  const [cols, setCols] = createSignal(4)
  const [rows, setRows] = createSignal(3)
  const [spacing, setSpacing] = createSignal(1.5)
  const [checkerboard, setCheckerboard] = createSignal(false)
  const [randomSeed, setRandomSeed] = createSignal(Date.now())

  const tagSize = 40

  const totalTags = createMemo(() => cols() * rows())
  const tagIds = createMemo(() => selectRandomTags(totalTags(), randomSeed()))

  const boardSize = createMemo(() => (spacing() - 1) * tagSize)
  const marginSize = createMemo(() => tagSize)
  const gridWidth = createMemo(() => cols() * tagSize + (cols() - 1) * boardSize())
  const gridHeight = createMemo(() => rows() * tagSize + (rows() - 1) * boardSize())
  const svgWidth = createMemo(() => gridWidth() + 2 * marginSize())
  const svgHeight = createMemo(() => gridHeight() + 2 * marginSize())
  const cellSize = createMemo(() => tagSize / 8) // 6x6 inner + 2 border cells = 8x8 total

  const tagPositions = createMemo(() => {
    const positions: { x: number; y: number; tagId: number }[] = []
    for (let r = 0; r < rows(); r++) {
      for (let c = 0; c < cols(); c++) {
        const idx = r * cols() + c
        const tagId = tagIds()[idx] ?? idx % TAG36H11_COUNT
        const x = marginSize() + c * (tagSize + boardSize())
        const y = marginSize() + r * (tagSize + boardSize())
        positions.push({ x, y, tagId })
      }
    }
    return positions
  })

  const checkerPositions = createMemo<{ x: number; y: number }[]>(() => {
    if (!checkerboard() || cols() < 2 || rows() < 2) {
      return []
    }
    const positions: { x: number; y: number }[] = []
    for (let r = 0; r < rows() - 1; r++) {
      for (let c = 0; c < cols() - 1; c++) {
        const x = marginSize() + (c + 1) * tagSize + c * boardSize()
        const y = marginSize() + (r + 1) * tagSize + r * boardSize()
        positions.push({ x, y })
      }
    }
    return positions
  })

  const svgContent = createMemo(() => {
    const w = svgWidth()
    const h = svgHeight()
    let content = `<rect width="${w}" height="${h}" fill="white"/>`

    for (const tag of tagPositions()) {
      const pattern = tagIdPattern(tag.tagId)
      for (let row = 0; row < 8; row++) {
        for (let col = 0; col < 8; col++) {
          // Border: outer ring (positions 0 or 7 in either dimension)
          const isBorder = row === 0 || row === 7 || col === 0 || col === 7
          const pRow = row - 1
          const pCol = col - 1
          let value = isBorder ? 0 : pattern[pRow * 6 + pCol]
          // background is white, only draw black modules
          if (value === 0) {
            const px = tag.x + col * cellSize()
            const py = tag.y + row * cellSize()
            content += `<rect x="${px}" y="${py}" width="${cellSize()}" height="${cellSize()}" fill="black"/>`
          }
        }
      }
    }

    for (const pos of checkerPositions()) {
      content += `<rect x="${pos.x}" y="${pos.y}" width="${boardSize()}" height="${boardSize()}" fill="black"/>`
    }

    return content
  })

  return (
    <div class={styles.root}>
      <div
        class={styles.display}
        innerHTML={`<svg xmlns="http://www.w3.org/2000/svg" width="${svgWidth()}" height="${svgHeight()}" viewBox="0 0 ${svgWidth()} ${svgHeight()}" style="width:100%;height:100%;shape-rendering:crispEdges">${svgContent()}</svg>`}
      />

      <aside class={styles.controls}>
        <h2 class={styles.title}>Target Settings</h2>

        <div class={styles.field}>
          <label class={styles.label}>Grid Size</label>
          <div class={styles.row}>
            <span class={styles.fieldLabel}>Columns</span>
            <select
              class={styles.select}
              value={String(cols())}
              onChange={(e) => setCols(parseInt(e.currentTarget.value))}
            >
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
              <option value="6">6</option>
            </select>
            <span class={styles.fieldLabel}>Rows</span>
            <select
              class={styles.select}
              value={String(rows())}
              onChange={(e) => setRows(parseInt(e.currentTarget.value))}
            >
              <option value="2">2</option>
              <option value="3">3</option>
              <option value="4">4</option>
              <option value="5">5</option>
              <option value="6">6</option>
            </select>
          </div>
        </div>

        <div class={styles.field}>
          <label class={styles.label}>Spacing</label>
          <select
            class={styles.select}
            value={String(spacing())}
            onChange={(e) => setSpacing(parseFloat(e.currentTarget.value))}
          >
            <option value="1">1x tag size</option>
            <option value="1.25">1.25x tag size</option>
            <option value="1.5">1.5x tag size</option>
            <option value="2">2x tag size</option>
          </select>
        </div>

        <div class={styles.field}>
          <label class={styles.checkboxLabel}>
            <input
              type="checkbox"
              checked={checkerboard()}
              onChange={(e) => setCheckerboard(e.currentTarget.checked)}
            />
            Include checkerboard between tags
          </label>
        </div>

        <button type="button" class={styles.randomizeBtn} onClick={() => setRandomSeed(Date.now())}>
          Randomize Tags
        </button>

        <div class={styles.field}>
          <span class={styles.label}>Expected IDs (same layout as target)</span>
          <div
            class={styles.legendList}
            style={{
              '--columns': cols(),
            }}
          >
            <For each={tagPositions()}>
              {(p) => (
                <div class={styles.legendCell} style={{ 'background-color': gridVizFillRgbCss(p().tagId) }}>
                  {p().tagId}
                </div>
              )}
            </For>
          </div>
        </div>

        <div class={styles.info}>
          <p>Press <span class={styles.kbd}>Ctrl + P</span> to print</p>
        </div>
      </aside>
    </div>
  )
}
