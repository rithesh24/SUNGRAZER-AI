import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import Starfield from './Starfield'
import { api } from '../lib/api'

const NAV = [
  { to: '/', label: 'Overview', end: true },
  { to: '/candidates', label: 'Candidates' },
  { to: '/review', label: 'Review queue' },
  { to: '/archaeology', label: 'Archaeology' },
  { to: '/unknown', label: 'Unknown objects' },
]

export default function Layout() {
  const [healthy, setHealthy] = useState<boolean | null>(null)

  useEffect(() => {
    api.health().then(() => setHealthy(true)).catch(() => setHealthy(false))
  }, [])

  return (
    <>
      <Starfield />
      <div className="shell">
        <header className="topbar">
          <NavLink to="/" className="brand">
            SUNGRAZER<span>&nbsp;AI</span>
          </NavLink>
          <nav className="nav">
            {NAV.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end}
                className={({ isActive }) => (isActive ? 'active' : '')}>
                {item.label}
              </NavLink>
            ))}
          </nav>
          <div className={`health ${healthy ? 'ok' : ''}`}>
            <span className="dot" />
            {healthy === null ? 'connecting' : healthy ? 'API online' : 'API offline'}
          </div>
        </header>
        <Outlet />
      </div>
    </>
  )
}
