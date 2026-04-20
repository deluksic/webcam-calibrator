# Solid.js 2.0 Patterns

## Async in createMemo

`createMemo` also accepts async functions. The memo value is of the resulting type; sync code does not continue while still pending (throws to stop execution).

```tsx
const data = createMemo(() => api.fetchData())

// In JSX - will suspend until resolved
<Show when={data()}>{(d) => <div>{d()}</div>}</Show>
```

## isPending Utility

Check if an async memo is still loading:

```tsx
const loading = () => isPending(() => someAsyncMemo())
```

## Signal Refs

Use `createSignal` + `ref` callback instead of `let` variables:

```tsx
const [el, setEl] = createSignal<HTMLDivElement>()

// In JSX
<div ref={setEl}>...</div>
```

**Wrong:**

```tsx
let el: HTMLDivElement
;<div ref={el}>...</div> // loses reactivity
```

## Error Boundaries

Use `<Errored>` at the app root instead of manual try/catch:

```tsx
import { Errored } from 'solid-js'

render(
  () => (
    <Errored fallback={(e) => <p class="error">{String(e)}</p>}>
      <App />
    </Errored>
  ),
  root,
)
```

Child components can throw and errors bubble up to the nearest Errored boundary.

## createStore - Functional Updaters

2.0 uses functional updaters instead of path-based setters:

```tsx
const [store, setStore] = createStore({ count: 0 })

// Old (v1):
setStore('count', 5)

// New (v2):
setStore((s) => {
  s.count = 5
})

// With nested objects:
setStore((s) => {
  s.user.name = 'Alice'
})
```

## Array Class Syntax

Replace `classList` with array class (like clsx):

```tsx
<button class={[styles.btn, isActive() && styles.active]}>
```

## `<For>`: item and index are accessors

In Solid 2.0, the render callback receives **accessors** (zero-arg functions), not raw values. Always call them. The second argument (`index`) is optional—omit it if you do not need it.

```tsx
<For each={items()}>{(item) => <span>{item().label}</span>}</For>
```

**Wrong:** `item.label` — stale / non-reactive.

## No /store Subpath

Import directly from `solid-js`:

```tsx
import { createStore } from 'solid-js'
```

## onCleanup

Import from `solid-js` (not `solid-js/web` for non-DOM utilities):

```tsx
import { onCleanup } from 'solid-js'
```

## createRoot for Effects at Module Level

If you need to set up effects/async outside of component hierarchy:

```tsx
import { createRoot } from 'solid-js'

createRoot(async (dispose) => {
  // async setup here
  onCleanup(() => dispose())
})
```

Note: With createMemo supporting async, this is often unnecessary now.

## Loading State with isPending

For loading UI while async memos resolve:

```tsx
<Show when={isPending(() => gpuRoot())}>
  <p>Loading...</p>
</Show>
```

## Key Differences from Solid.js 1.x

1. `createMemo` handles async natively
2. `createStore` uses functional updaters
3. No `classList` - use array class syntax
4. No `/store` subpath for imports
5. `isPending()` utility for async state
6. `<Errored>` for error boundaries (replaces try/catch) when exported by your Solid build
7. `flush()` to manually flush pending updates
8. `<For>` / `<Index>` pass **accessors** for item (and index where applicable) — call `item()`, `index()`
