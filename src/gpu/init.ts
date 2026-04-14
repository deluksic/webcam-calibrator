// TypeGPU initialization — singleton GPU root
import * as tgpu from 'typegpu';

let root: Awaited<ReturnType<typeof tgpu.init>> | null = null;

export async function initGPU() {
  if (root) return root;

  if (!navigator.gpu) {
    throw new Error(
      'WebGPU is not supported. Please use Chrome 113+, Edge 113+, ' +
      'or Firefox Nightly with WebGPU enabled.'
    );
  }

  root = await tgpu.init();
  return root;
}

export function getRoot() {
  if (!root) throw new Error('GPU not initialized — call initGPU() first');
  return root;
}
