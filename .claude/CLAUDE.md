# webcam-calibrator skills

## Ship
`pnpm ship` — builds and copies `dist/*` to Apache docroot.
⚠️ Do NOT run `pnpm build` before ship — ship includes its own build.

## SolidJS 2.0

- CSS style props use kebab-case strings: `'margin-top'` NOT `marginTop`
- Use `class` not `className`
- Signals accessed as functions: `displayMode()` not `displayMode`
- Setter pattern: `setSignal(prev => newValue)` for derived updates
- `createMemo(fn)` is eager by default

## TypeGPU WGSL Limitations

### Cannot Unroll Loop Containing `continue`

**Error**: `Cannot unroll loop containing continue`

TypeGPU's `std.range()` + `tgpu.unroll()` cannot unroll loops that contain `continue` statements.

**Workaround**: Replace `continue` with conditional logic that skips processing:

```typescript
// ❌ Bad - cannot unroll
for (const ix of tgpu.unroll(std.range(3))) {
  if (skipCondition) { continue; }
  // ...
}

// ✅ Good - conditional write
for (const ix of tgpu.unroll(std.range(3))) {
  if (!skipCondition) {
    // ... process
  }
}
```