import * as Comlink from 'comlink'

import type { IdentityDesignSpec, IdentityRunOutput } from '@/lib/pointIdentity/identityPipeline'

import type { IdentityJob, IdentityWorkerApi } from './identity.worker'

let worker: Worker | undefined
let proxy: Comlink.Remote<IdentityWorkerApi> | undefined

function remote(): Comlink.Remote<IdentityWorkerApi> {
  if (proxy === undefined) {
    worker = new Worker(new URL('./identity.worker.ts', import.meta.url), { type: 'module' })
    proxy = Comlink.wrap<IdentityWorkerApi>(worker)
  }
  return proxy
}

/** Rebuild the worker's design cache ahead of the first frame that needs it. */
export function configureIdentityWorker(spec: IdentityDesignSpec): void {
  void Promise.resolve(remote().configure(spec)).catch((cause: unknown) => {
    console.error('[identity-worker] configure failed', cause)
  })
}

export function runIdentityInWorker(
  spec: IdentityDesignSpec,
  job: IdentityJob,
): Promise<IdentityRunOutput> {
  return remote().run(spec, job)
}

export function disposeIdentityWorker(): void {
  worker?.terminate()
  worker = undefined
  proxy = undefined
}
