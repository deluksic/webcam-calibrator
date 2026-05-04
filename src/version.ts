import packageJson from '../package.json' with { type: 'json' }

function getEntryChunkHashFromScript(): string | undefined {
  const script = document.querySelector<HTMLScriptElement>('script[type="module"][src*="index-"]')
  if (!script?.src) {
    return undefined
  }

  const scriptFile = script.src.split('/').pop()?.split('?')[0] ?? ''
  const match = scriptFile.match(/^index-([^.]+)\./)
  return match?.[1]
}

const hash = getEntryChunkHashFromScript()
export const VERSION: string = hash !== undefined ? `${packageJson.version}-${hash}` : 'dev'
console.log(`[build] ${VERSION}`)
