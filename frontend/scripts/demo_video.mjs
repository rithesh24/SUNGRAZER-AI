// Records a ~90s scripted walkthrough of all six views as a .webm.
// Usage: node scripts/demo_video.mjs   (needs uvicorn :8000 + vite :5173)
// Output: ../docs/demo/sungrazer-demo.webm
import { chromium } from 'playwright'
import { mkdirSync, renameSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const BASE = 'http://localhost:5173'
const OUT = fileURLToPath(new URL('../../docs/demo/', import.meta.url))
const COMET = 'seq18_e91e65277cbe11c1_00070'

mkdirSync(OUT, { recursive: true })
const browser = await chromium.launch()
const context = await browser.newContext({
  viewport: { width: 1600, height: 1000 },
  recordVideo: { dir: OUT, size: { width: 1600, height: 1000 } },
})
const page = await context.newPage()
const pause = (ms) => page.waitForTimeout(ms)

async function scrollThrough(steps, stepPx = 500, ms = 900) {
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, stepPx)
    await pause(ms)
  }
}

// 1. Overview — let the starfield breathe, then tour the numbers.
await page.goto(BASE + '/')
await page.waitForSelector('.tile .value')
await pause(4000)
await scrollThrough(3)
await pause(1500)

// 2. Explorer — ranked list, filter to high priority.
await page.goto(BASE + '/candidates')
await page.waitForSelector('tbody tr')
await pause(3000)
await page.selectOption('.filters select', 'HIGH_PRIORITY')
await page.waitForSelector('tbody tr')
await pause(3500)

// 3. Detail — the known comet: crops animate, report streams in.
await page.goto(BASE + `/candidates/${COMET}`)
await page.waitForSelector('.crop-frame')
await pause(9000) // flipbook loops + agent report arrives
await scrollThrough(4)
await pause(2000)

// 4. Review queue.
await page.goto(BASE + '/review')
await page.waitForSelector('tbody tr')
await pause(3500)

// 5. Archaeology — DNA search on the comet: family appears.
await page.goto(BASE + `/archaeology?track=${COMET}`)
await page.waitForSelector('tbody tr')
await pause(5000)
await scrollThrough(2)

// 6. Unknown objects, then rest on overview.
await page.goto(BASE + '/unknown')
await page.waitForSelector('tbody tr')
await pause(3000)
await page.goto(BASE + '/')
await page.waitForSelector('.tile .value')
await pause(4000)

await context.close() // flushes the video
await browser.close()
const video = readdirSync(OUT).find((f) => f.endsWith('.webm'))
renameSync(OUT + video, OUT + 'sungrazer-demo.webm')
console.log('saved docs/demo/sungrazer-demo.webm')
