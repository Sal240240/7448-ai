import { useEffect, useState } from 'react'
import { Home } from './pages/Home'
import { InsideModel } from './pages/InsideModel'
import { Methodology } from './pages/Methodology'
import { Outlook } from './pages/Outlook'
import { Simulator } from './pages/Simulator'

const PAGES = [
  { id: 'home', label: 'Overview' },
  { id: 'simulator', label: 'Simulator' },
  { id: 'outlook', label: 'Outlook' },
  { id: 'model', label: 'Inside the model' },
  { id: 'methodology', label: 'Methodology' },
] as const

type PageId = (typeof PAGES)[number]['id']

function currentPage(): PageId {
  const hash = window.location.hash.replace('#', '')
  return (PAGES.find((p) => p.id === hash)?.id ?? 'home') as PageId
}

/**
 * Hash routing rather than a router dependency: four static pages, no nested
 * routes, no data loading tied to URL params. Adding react-router here would be
 * a dependency that buys nothing.
 */
export function App() {
  const [page, setPage] = useState<PageId>(currentPage)

  useEffect(() => {
    const onHashChange = () => {
      setPage(currentPage())
      window.scrollTo({ top: 0 })
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  const navigate = (id: string) => {
    window.location.hash = id
  }

  return (
    <div className="shell">
      <header className="nav">
        <div className="wrap nav__inner">
          <a className="nav__mark" href="#home">
            7448&thinsp;AI
          </a>
          <nav className="nav__links" aria-label="Main">
            {PAGES.map((p) => (
              <button
                key={p.id}
                className="nav__link"
                aria-current={page === p.id ? 'page' : undefined}
                onClick={() => navigate(p.id)}
              >
                {p.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main>
        {page === 'home' && <Home onNavigate={navigate} />}
        {page === 'simulator' && <Simulator />}
        {page === 'outlook' && <Outlook />}
        {page === 'model' && <InsideModel />}
        {page === 'methodology' && <Methodology />}
      </main>

      <footer className="footer">
        <div className="wrap">
          <div className="row" style={{ justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <span>
              7448 AI — predicting gut metabolite profiles from microbiome composition. Research
              tool; not medical advice.
            </span>
            <span className="tiny">
              Built on public data from the Borenstein Lab collection, Disbiome, KEGG and PubMed.
            </span>
          </div>
        </div>
      </footer>
    </div>
  )
}
