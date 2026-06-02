import React from 'react'
import { createRoot } from 'react-dom/client'
import 'driver.js/dist/driver.css'
import App from './App'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
