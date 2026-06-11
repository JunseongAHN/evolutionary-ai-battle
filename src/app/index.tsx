import React from 'react';
import { createRoot } from 'react-dom/client';
import { SimulationPage } from '../pages/simulation/view';

const rootElement = document.getElementById('evolutionary-ai-battle');
if (!rootElement) {
    throw new Error('Simulation page root #evolutionary-ai-battle was not found');
}

createRoot(rootElement).render(<SimulationPage />);
