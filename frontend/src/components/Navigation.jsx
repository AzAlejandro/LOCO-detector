import React, { useState } from 'react'

/**
 * Navigation component: collapsible sidebar (level 1) + exported GROUPS for tab strip.
 *
 * Props:
 *   activeGroup: 'scribble' | 'loco' | 'detection'
 *   activeTab: string
 *   onGroupChange: (group) => void
 *   onTabChange: (tab) => void
 */

const GROUPS = [
  {
    key: 'scribble',
    label: 'Scribble',
    fullLabel: 'Grupo 1: Entrenamiento Scribble',
    icon: '✏️',
    section: 'training',
    tabs: [
      { key: 'workbench', label: 'Scribbles y Experimentos' },
      { key: 'review', label: 'Revision de Resultados' },
      { key: 'models', label: 'Modelos de Asistencia' },
    ],
  },
  {
    key: 'loco',
    label: 'LOCO',
    fullLabel: 'Grupo 2: Entrenamiento LOCO',
    icon: '🔵',
    section: 'training',
    tabs: [
      { key: 'locoDataset', label: 'Generar Dataset' },
      { key: 'locoAugment', label: 'Aumentacion' },
      { key: 'locoTraining', label: 'Entrenamiento' },
      { key: 'locoTest', label: 'Test de Modelo' },
    ],
  },
  {
    key: 'detection',
    label: 'Deteccion',
    fullLabel: 'Grupo 3: Deteccion y Medicion',
    icon: '📏',
    section: 'production',
    tabs: [
      { key: 'locoModel', label: 'Detector LOCO' },
      { key: 'diameter', label: 'Medicion de Diametros' },
    ],
  },
]

export function legacyToGroup(tab) {
  const mapping = {
    workbench: { group: 'scribble', tab: 'workbench' },
    review: { group: 'scribble', tab: 'review' },
    models: { group: 'scribble', tab: 'models' },
    locoDataset: { group: 'loco', tab: 'locoDataset' },
    locoAugment: { group: 'loco', tab: 'locoAugment' },
    locoTraining: { group: 'loco', tab: 'locoTraining' },
    locoTest: { group: 'loco', tab: 'locoTest' },
    locoModel: { group: 'detection', tab: 'locoModel' },
    diameter: { group: 'detection', tab: 'diameter' },
    loco: { group: 'detection', tab: 'locoModel' },
  }
  return mapping[tab] || { group: 'scribble', tab: 'workbench' }
}

export function groupToLegacy(group, tab) {
  const mapping = {
    'scribble:workbench': 'workbench',
    'scribble:review': 'review',
    'scribble:models': 'models',
    'loco:locoDataset': 'locoDataset',
    'loco:locoAugment': 'locoAugment',
    'loco:locoTraining': 'locoTraining',
    'loco:locoTest': 'locoTest',
    'detection:locoModel': 'locoModel',
    'detection:diameter': 'diameter',
  }
  return mapping[`${group}:${tab}`] || 'workbench'
}

export { GROUPS }

export default function Navigation({ activeGroup, onGroupChange, onTabChange }) {
  const [collapsed, setCollapsed] = useState(true)

  function handleGroupClick(key) {
    const group = GROUPS.find((item) => item.key === key)
    const firstTab = group ? group.tabs[0].key : 'workbench'
    onGroupChange(key)
    onTabChange(firstTab)
  }

  return (
    <aside
      className={`sidebar ${collapsed ? 'collapsed' : 'expanded'}`}
      onMouseEnter={() => setCollapsed(false)}
      onMouseLeave={() => setCollapsed(true)}
    >
      <button
        className="sidebar-toggle"
        onClick={() => setCollapsed((current) => !current)}
        title={collapsed ? 'Expandir menu' : 'Colapsar menu'}
      >
        <span className="sidebar-toggle-icon">{collapsed ? '☰' : '✕'}</span>
        {!collapsed && <span className="sidebar-toggle-label">Menu</span>}
      </button>

      <div className="sidebar-section-label">{collapsed ? '⚙' : 'Entrenamiento'}</div>
      {GROUPS.filter((group) => group.section === 'training').map((group) => (
        <button
          key={group.key}
          className={`sidebar-group-btn ${activeGroup === group.key ? 'active' : ''}`}
          onClick={() => handleGroupClick(group.key)}
          title={collapsed ? group.fullLabel : group.label}
        >
          <span className="sidebar-icon">{group.icon}</span>
          {!collapsed && <span className="sidebar-label">{group.label}</span>}
        </button>
      ))}

      <div className="sidebar-separator" />
      <div className="sidebar-section-label">{collapsed ? '⚙' : 'Produccion'}</div>
      {GROUPS.filter((group) => group.section === 'production').map((group) => (
        <button
          key={group.key}
          className={`sidebar-group-btn ${activeGroup === group.key ? 'active' : ''}`}
          onClick={() => handleGroupClick(group.key)}
          title={collapsed ? group.fullLabel : group.label}
        >
          <span className="sidebar-icon">{group.icon}</span>
          {!collapsed && <span className="sidebar-label">{group.label}</span>}
        </button>
      ))}
    </aside>
  )
}
