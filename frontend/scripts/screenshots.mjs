// Demo screenshot capture: all six views against the running dev stack.
// Usage: node scripts/screenshots.mjs   (needs uvicorn :8000 + vite :5173)
// Output: ../docs/screenshots/*.png
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const BASE = 'http://localhost:5173'
const OUT = fileURLToPath(new URL('../../docs/screenshots/', import.meta.url))
const COMET = 'seq18_e91e65277cbe11c1_00070' // SOHO-5053, global rank #1

const SHOTS = [
  { name: '1-overview', path: '/', waitFor: '.tile .value' },
  { name: '2-explorer', path: '/candidates', waitFor: 'tbody tr' },
  { name: '3-detail', path: `/candidates/${COMET}`, waitFor: '.crop-frame', settle: 12000 },
  { name: '4-review-queue', path: '/review', waitFor: 'tbody tr' },
  { name: '5-archaeology', path: `/archaeology?track=${COMET}`, waitFor: 'tbody tr' },
  { name: '6-unknown', path: '/unknown', waitFor: 'tbody tr' },
]

mkdirSync(OUT, { recursive: true })
const browser = await chromium.launch()
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } })

for (const shot of SHOTS) {
  await page.goto(BASE + shot.path)
  await page.waitForSelector(shot.waitFor, { timeout: 30000 })
  // settle: let the agent report / starfield / images finish rendering
  await page.waitForTimeout(shot.settle ?? 2500)
  await page.screenshot({ path: `${OUT}${shot.name}.png`, fullPage: shot.name === '3-detail' })
  console.log(`captured ${shot.name}`)
}

await browser.close()
