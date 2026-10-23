import { verifyModelAsset } from './modelAsset'

verifyModelAsset()
  .then((result) => {
    console.log(`[model] verified ${result.path} (${result.bytes} bytes)`)
  })
  .catch((cause) => {
    console.error(`[model] ${cause instanceof Error ? cause.message : String(cause)}`)
    process.exitCode = 1
  })
