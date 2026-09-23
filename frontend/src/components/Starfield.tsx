import { useEffect, useRef } from 'react'

// Full-viewport animated starfield: three parallax depth layers that drift
// slowly and follow the mouse, a soft milky-way band, and the occasional
// comet streak. Pauses when the tab is hidden; renders a single static
// frame when the user prefers reduced motion.

interface Star {
  x: number // 0..1 of viewport
  y: number
  z: number // depth 0 (far) .. 1 (near)
  r: number
  hue: string
  twinkle: number // phase
}

const STAR_COLORS = ['#ffffff', '#cfe2ff', '#ffe9c9', '#9fb6ff']
const STAR_RGB: Record<string, string> = {
  '#ffffff': '255,255,255',
  '#cfe2ff': '207,226,255',
  '#ffe9c9': '255,233,201',
  '#9fb6ff': '159,182,255',
}

// Soft radial glow sprite, one per star color, drawn around bright stars.
function makeGlowSprites(): Record<string, HTMLCanvasElement> {
  const sprites: Record<string, HTMLCanvasElement> = {}
  for (const [hex, rgb] of Object.entries(STAR_RGB)) {
    const off = document.createElement('canvas')
    off.width = off.height = 32
    const g = off.getContext('2d')!
    const grad = g.createRadialGradient(16, 16, 0, 16, 16, 16)
    grad.addColorStop(0, `rgba(${rgb},0.9)`)
    grad.addColorStop(0.3, `rgba(${rgb},0.25)`)
    grad.addColorStop(1, `rgba(${rgb},0)`)
    g.fillStyle = grad
    g.fillRect(0, 0, 32, 32)
    sprites[hex] = off
  }
  return sprites
}

function makeStars(count: number): Star[] {
  const stars: Star[] = []
  for (let i = 0; i < count; i++) {
    const z = Math.random()
    stars.push({
      x: Math.random(),
      y: Math.random(),
      z,
      r: 0.4 + z * 1.4 + Math.random() * 0.6,
      hue: STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)],
      twinkle: Math.random() * Math.PI * 2,
    })
  }
  return stars
}

interface Comet {
  x: number
  y: number
  vx: number
  vy: number
  life: number // frames remaining
}

function paintMilkyWay(width: number, height: number): HTMLCanvasElement {
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const g = off.getContext('2d')!
  g.translate(width / 2, height / 2)
  g.rotate(-0.5)
  // Layered soft ellipses along one diagonal band.
  const blobs = 26
  for (let i = 0; i < blobs; i++) {
    const t = i / blobs - 0.5
    const cx = t * width * 1.5
    const cy = Math.sin(t * 5) * height * 0.05
    const radius = height * (0.12 + Math.random() * 0.16)
    const grad = g.createRadialGradient(cx, cy, 0, cx, cy, radius)
    const tint = i % 4 === 0 ? '110,140,220' : i % 3 === 0 ? '150,120,200' : '190,205,235'
    grad.addColorStop(0, `rgba(${tint},${0.05 + Math.random() * 0.05})`)
    grad.addColorStop(1, 'rgba(0,0,0,0)')
    g.fillStyle = grad
    g.beginPath()
    g.arc(cx, cy, radius, 0, Math.PI * 2)
    g.fill()
  }
  return off
}

export default function Starfield() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current!
    const ctx = canvas.getContext('2d')!
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const stars = makeStars(420)
    const comets: Comet[] = []
    const glows = makeGlowSprites()
    let milkyWay: HTMLCanvasElement | null = null
    let raf = 0
    let time = 0
    let width = 0
    let height = 0
    // Mouse parallax: target follows the pointer, offset eases toward it.
    let targetX = 0
    let targetY = 0
    let offsetX = 0
    let offsetY = 0

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = window.innerWidth
      height = window.innerHeight
      canvas.width = width * dpr
      canvas.height = height * dpr
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      milkyWay = paintMilkyWay(width, height)
      if (reduced) frame(true)
    }

    function onMouse(e: MouseEvent) {
      targetX = e.clientX / width - 0.5
      targetY = e.clientY / height - 0.5
    }

    function frame(once = false) {
      time += 1
      offsetX += (targetX - offsetX) * 0.03
      offsetY += (targetY - offsetY) * 0.03

      ctx.clearRect(0, 0, width, height)
      if (milkyWay) {
        // The band itself parallaxes least (farthest away).
        ctx.globalAlpha = 1
        ctx.drawImage(milkyWay, -offsetX * 12, -offsetY * 12)
      }

      const drift = time * 0.00001
      for (const s of stars) {
        const parallax = 14 + s.z * 42
        const x = (((s.x + drift * (0.4 + s.z)) % 1) + 1) % 1
        const px = x * width - offsetX * parallax
        const py = s.y * height - offsetY * parallax
        const tw = 0.65 + 0.35 * Math.sin(s.twinkle + time * (0.01 + s.z * 0.02))
        ctx.globalAlpha = (0.35 + s.z * 0.6) * tw
        ctx.fillStyle = s.hue
        ctx.beginPath()
        ctx.arc(px, py, s.r, 0, Math.PI * 2)
        ctx.fill()
        if (s.z > 0.55) {
          // Bright (near) stars get a soft glow halo.
          const halo = s.r * 7
          ctx.globalAlpha = (0.35 + s.z * 0.35) * tw
          ctx.drawImage(glows[s.hue], px - halo / 2, py - halo / 2, halo, halo)
        }
      }

      // Comet shower: a group of 2-3 on parallel paths (shared velocity, so
      // they never cross); the next group spawns only after all are gone.
      if (!once && comets.length === 0 && Math.random() < 1 / 120) {
        const fromLeft = Math.random() < 0.5
        const vx = (fromLeft ? 1 : -1) * (4 + Math.random() * 3)
        const vy = 1.2 + Math.random() * 1.4
        const n = 2 + Math.floor(Math.random() * 2)
        for (let i = 0; i < n; i++) {
          // Trailing comets start staggered behind the leader.
          const lag = i * (20 + Math.random() * 25)
          comets.push({
            x: (fromLeft ? -60 : width + 60) - vx * lag,
            y: Math.random() * height * 0.5,
            vx,
            vy,
            life: 260 + lag,
          })
        }
      }
      for (let i = comets.length - 1; i >= 0; i--) {
        const c = comets[i]
        c.x += c.vx
        c.y += c.vy
        c.life -= 1
        const fade = Math.min(1, c.life / 60)
        const tail = 26
        const grad = ctx.createLinearGradient(c.x, c.y, c.x - c.vx * tail, c.y - c.vy * tail)
        grad.addColorStop(0, `rgba(210,230,255,${0.9 * fade})`)
        grad.addColorStop(1, 'rgba(210,230,255,0)')
        ctx.strokeStyle = grad
        ctx.lineWidth = 1.6
        ctx.globalAlpha = 1
        ctx.beginPath()
        ctx.moveTo(c.x, c.y)
        ctx.lineTo(c.x - c.vx * tail, c.y - c.vy * tail)
        ctx.stroke()
        // Cull only past the far edge (trailing group members spawn deep
        // behind the near edge and must survive their approach).
        const gone = c.vx > 0 ? c.x > width + 100 : c.x < -100
        if (c.life <= 0 || gone || c.y > height + 100) {
          comets.splice(i, 1)
        }
      }

      ctx.globalAlpha = 1
      if (!once) raf = requestAnimationFrame(() => frame())
    }

    function onVisibility() {
      cancelAnimationFrame(raf)
      if (!document.hidden && !reduced) raf = requestAnimationFrame(() => frame())
    }

    resize()
    window.addEventListener('resize', resize)
    if (!reduced) {
      window.addEventListener('mousemove', onMouse)
      document.addEventListener('visibilitychange', onVisibility)
      raf = requestAnimationFrame(() => frame())
    }
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMouse)
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  return <canvas ref={canvasRef} className="starfield" aria-hidden="true" />
}
