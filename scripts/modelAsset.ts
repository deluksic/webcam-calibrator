import { createHash } from 'node:crypto'
import { mkdir, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
export const MODEL_MANIFEST_PATH = path.join(REPO_ROOT, 'model-manifest.json')

export type ModelManifest = {
  version: number
  releaseTag: string
  url: string
  path: string
  bytes?: number
  sha256: string
}

export type EnsureModelAssetOptions = {
  force?: boolean
  log?: (message: string) => void
}

export type EnsureModelAssetResult = {
  path: string
  sha256: string
  bytes: number
  fetched: boolean
}

function sha256(data: Uint8Array): string {
  return createHash('sha256').update(data).digest('hex')
}

function validateManifest(value: unknown): ModelManifest {
  if (typeof value !== 'object' || value === null) {
    throw new Error('model-manifest.json must be an object')
  }
  const raw = value as Record<string, unknown>
  if (typeof raw.url !== 'string' || raw.url.length === 0) {
    throw new Error('model-manifest.json: `url` must be a non-empty string')
  }
  if (typeof raw.path !== 'string' || raw.path.length === 0) {
    throw new Error('model-manifest.json: `path` must be a non-empty string')
  }
  if (typeof raw.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(raw.sha256)) {
    throw new Error('model-manifest.json: `sha256` must be a 64-character hex string')
  }
  return {
    version: typeof raw.version === 'number' ? raw.version : 1,
    releaseTag: typeof raw.releaseTag === 'string' ? raw.releaseTag : '',
    url: raw.url,
    path: raw.path,
    bytes: typeof raw.bytes === 'number' ? raw.bytes : undefined,
    sha256: raw.sha256.toLowerCase(),
  }
}

export async function readModelManifest(): Promise<ModelManifest> {
  const text = await readFile(MODEL_MANIFEST_PATH, 'utf8')
  return validateManifest(JSON.parse(text))
}

async function readExisting(destination: string): Promise<Uint8Array | undefined> {
  try {
    const info = await stat(destination)
    if (!info.isFile() || info.size === 0) {
      return undefined
    }
    return await readFile(destination)
  } catch (cause) {
    if ((cause as NodeJS.ErrnoException).code === 'ENOENT') {
      return undefined
    }
    throw cause
  }
}

async function downloadAsset(manifest: ModelManifest, log: (message: string) => void): Promise<Uint8Array> {
  const source = process.env.MODEL_ASSET_URL?.trim() || manifest.url
  log(`fetching ${source}`)
  if (source.startsWith('file:')) {
    return readFile(fileURLToPath(source))
  }
  if (!/^https?:\/\//i.test(source)) {
    return readFile(path.resolve(REPO_ROOT, source))
  }
  const response = await fetch(source, { redirect: 'follow' })
  if (!response.ok) {
    throw new Error(
      `Failed to download model weights (${response.status} ${response.statusText}) from ${source}. ` +
        `Create the ${manifest.releaseTag || 'model'} GitHub release or set MODEL_ASSET_URL.`,
    )
  }
  return new Uint8Array(await response.arrayBuffer())
}

/**
 * Ensure the ignored `public/models` asset exists and matches the manifest.
 * Used by the Vite dev server, the production build plugin, and CI.
 */
export async function ensureModelAsset(options: EnsureModelAssetOptions = {}): Promise<EnsureModelAssetResult> {
  const log = options.log ?? (() => undefined)
  const manifest = await readModelManifest()
  const destination = path.resolve(REPO_ROOT, manifest.path)
  const expectedHash = manifest.sha256

  if (!options.force) {
    const existing = await readExisting(destination)
    if (existing && sha256(existing) === expectedHash) {
      log(`model weights OK (${existing.byteLength} bytes, ${expectedHash.slice(0, 12)}…)`)
      return {
        path: destination,
        sha256: expectedHash,
        bytes: existing.byteLength,
        fetched: false,
      }
    }
  }

  const bytes = await downloadAsset(manifest, log)
  const actualHash = sha256(bytes)
  if (actualHash !== expectedHash) {
    throw new Error(
      `Model checksum mismatch: expected ${expectedHash}, got ${actualHash}. ` +
        `Update model-manifest.json and the release asset together.`,
    )
  }
  if (manifest.bytes !== undefined && bytes.byteLength !== manifest.bytes) {
    throw new Error(`Model size mismatch: expected ${manifest.bytes} bytes, got ${bytes.byteLength}.`)
  }

  await mkdir(path.dirname(destination), { recursive: true })
  const temporary = `${destination}.tmp-${process.pid}`
  try {
    await writeFile(temporary, bytes)
    await rename(temporary, destination)
  } catch (cause) {
    await unlink(temporary).catch(() => undefined)
    throw cause
  }
  log(`model weights downloaded (${bytes.byteLength} bytes, ${actualHash.slice(0, 12)}…)`)
  return { path: destination, sha256: actualHash, bytes: bytes.byteLength, fetched: true }
}

/** Verify the existing ignored asset without network access. */
export async function verifyModelAsset(): Promise<EnsureModelAssetResult> {
  const manifest = await readModelManifest()
  const destination = path.resolve(REPO_ROOT, manifest.path)
  const existing = await readExisting(destination)
  if (!existing) {
    throw new Error(
      `Missing ${manifest.path}. Run \`pnpm model:fetch\` or download the ${manifest.releaseTag || 'model'} release asset.`,
    )
  }
  const actualHash = sha256(existing)
  if (actualHash !== manifest.sha256) {
    throw new Error(
      `Model checksum mismatch for ${manifest.path}: expected ${manifest.sha256}, got ${actualHash}.`,
    )
  }
  return { path: destination, sha256: actualHash, bytes: existing.byteLength, fetched: false }
}
