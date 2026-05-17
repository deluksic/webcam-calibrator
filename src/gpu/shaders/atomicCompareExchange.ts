import tgpu, { d } from 'typegpu'

/** Matches WGSL `atomicCompareExchangeResult<i32>`. */
export const AtomicCompareExchangeResultI32 = d.struct({
  old_value: d.i32,
  exchanged: d.bool,
})

/**
 * `atomicCompareExchangeWeak` for `atomic<i32>` storage (not in typegpu/std yet).
 * @see typegpu/src/core/rawCodeSnippet/tgpuRawCodeSnippet.ts
 */
export const atomicCompareExchangeWeakI32 = tgpu.fn(
  [d.ptrStorage(d.atomic(d.i32), 'read-write'), d.i32, d.i32],
  AtomicCompareExchangeResultI32,
)`
  (ptr, compare, value) -> AtomicCompareExchangeResultI32 {
    let res = atomicCompareExchangeWeak(ptr, compare, value);
    return AtomicCompareExchangeResultI32(res.old_value, res.exchanged);
  }
`
