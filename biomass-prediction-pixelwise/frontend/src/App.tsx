import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import MapExplorerPage from './pages/MapExplorerPage';
import MLModelPage from './pages/MLModelPage';
import MLModelPageWithRegions from './pages/MLModelPage_RegionSelection';
import UrbanAGBPage from './pages/UrbanAGBPage';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/map" element={<MapExplorerPage />} />
          <Route path="/model" element={<MLModelPage />} />
          <Route path="/model-regions" element={<MLModelPageWithRegions />} />
          <Route path="/urban-agb" element={<UrbanAGBPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;