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
        ctx.globalAlpha = (0.25 + s.z * 0.6) * tw
        ctx.fillStyle = s.hue
        ctx.beginPath()
        ctx.arc(px, py, s.r, 0, Math.PI * 2)
        ctx.fill()
      }

      // Rare comet streak: ~ every 12s at 60fps.
      if (!once && comets.length === 0 && Math.random() < 1 / 720) {
        const fromLeft = Math.random() < 0.5
        comets.push({
          x: fromLeft ? -60 : width + 60,
          y: Math.random() * height * 0.5,
          vx: (fromLeft ? 1 : -1) * (4 + Math.random() * 3),
          vy: 1.2 + Math.random() * 1.4,
          life: 260,
        })
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
        if (c.life <= 0 || c.x < -100 || c.x > width + 100 || c.y > height + 100) {
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
