import typegpu from 'eslint-plugin-typegpu'
import { defineConfig } from 'oxlint'

export default defineConfig({
  categories: {
    correctness: 'error',
  },
  jsPlugins: ['eslint-plugin-typegpu'],
  rules: {
    ...typegpu.configs.recommended.rules,
    // (optional) changes from the recommended preset
  },
})
