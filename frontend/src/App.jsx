import { Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/Layout'
import HomePage from './pages/HomePage'
import AnalysisPage from './pages/AnalysisPage'
import ModelsPage from './pages/ModelsPage'
import DataPage from './pages/DataPage'
import AboutPage from './pages/AboutPage'

function App() {
  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          className: 'bg-space-800 text-white border border-space-700',
          duration: 4000,
        }}
      />
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<HomePage />} />
          <Route path="analysis" element={<AnalysisPage />} />
          <Route path="models" element={<ModelsPage />} />
          <Route path="data" element={<DataPage />} />
          <Route path="about" element={<AboutPage />} />
        </Route>
      </Routes>
    </>
  )
}

export default App
