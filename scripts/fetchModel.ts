import { ensureModelAsset } from './modelAsset'

const force = process.argv.includes('--force')

ensureModelAsset({ force, log: (message) => console.log(`[model] ${message}`) }).catch((cause) => {
  console.error(`[model] ${cause instanceof Error ? cause.message : String(cause)}`)
  process.exitCode = 1
})
