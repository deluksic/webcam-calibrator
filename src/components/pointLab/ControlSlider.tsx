import type { Component } from 'solid-js'

import styles from '@/components/pointLab/PointLabView.module.css'

type ControlSliderProps = {
  id: string
  label: string
  value: number
  min: number
  max: number
  step: number
  format?: (value: number) => string
  onInput: (value: number) => void
}

export const ControlSlider: Component<ControlSliderProps> = (props) => {
  const displayValue = () => (props.format === undefined ? String(props.value) : props.format(props.value))

  return (
    <div class={styles.control}>
      <div class={styles.controlHead}>
        <label class={styles.controlLabel} for={props.id}>
          {props.label}
        </label>
        <span class={styles.controlValue}>{displayValue()}</span>
      </div>
      <input
        id={props.id}
        class={styles.range}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onInput={(event) => props.onInput(Number(event.currentTarget.value))}
      />
    </div>
  )
}
