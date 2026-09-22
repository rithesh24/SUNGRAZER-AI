import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import Overview from './pages/Overview'
import Explorer from './pages/Explorer'
import Detail from './pages/Detail'
import Archaeology from './pages/Archaeology'

function Placeholder({ title, note }: { title: string; note: string }) {
  return (
    <>
      <h1 className="page-title">{title}</h1>
      <p className="page-sub">{note}</p>
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Overview />} />
          <Route path="candidates" element={<Explorer />} />
          <Route path="candidates/:trackId" element={<Detail />} />
          <Route path="review"
            element={<Explorer preset={{
              title: 'Review queue',
              sub: 'High-priority tracks awaiting human review, best first.',
              status: 'HIGH_PRIORITY', lockStatus: true,
            }} />} />
          <Route path="archaeology" element={<Archaeology />} />
          <Route path="unknown"
            element={<Explorer preset={{
              title: 'Unknown objects',
              sub: 'Tracks on excluded event days, kept out of automatic prioritization. Preserved, not hidden.',
              status: 'UNRESOLVED', lockStatus: true,
            }} />} />
          <Route path="*" element={<Placeholder title="Not found" note="No such view." />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
