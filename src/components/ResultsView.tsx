import styles from './ResultsView.module.css';

export function ResultsView() {
  return (
    <div class={styles.root}>
      <div class={styles.placeholder}>
        <p class={styles.placeholderText}>Calibration results coming soon</p>
      </div>
    </div>
  );
}