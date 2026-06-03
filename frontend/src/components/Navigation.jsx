import React, { useState } from 'react'

/**
 * Navigation component: collapsible sidebar (level 1) + exported GROUPS for tab strip.
 *
 * Props:
 *   activeGroup: 'scribble' | 'loco' | 'detection' | 'configuration' | 'tutorial'
 *   activeTab: string
 *   onGroupChange: (group) => void
 *   onTabChange: (tab) => void
 *   forceExpanded?: boolean
 */

const GROUPS = [
  {
    key: 'scribble',
    label: 'Scribble',
    fullLabel: 'Grupo 1: Entrenamiento Scribble',
    icon: 'edit_note',
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
    icon: 'adjust',
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
    icon: 'straighten',
    section: 'production',
    tabs: [
      { key: 'locoModel', label: 'Detector LOCO' },
      { key: 'diameter', label: 'Medicion de Diametros' },
    ],
  },
  {
    key: 'configuration',
    label: 'Configuracion',
    fullLabel: 'Configuracion del proyecto',
    icon: 'settings',
    section: 'other',
    tabs: [
      { key: 'projectTransfer', label: 'Exportar e importar' },
    ],
  },
  {
    key: 'tutorial',
    label: 'Tutorial',
    fullLabel: 'Tutoriales guiados del proyecto',
    icon: 'menu_book',
    section: 'other',
    tabs: [
      { key: 'tutorialOverview', label: 'General' },
      { key: 'tutorialScribble', label: 'Ruta Scribble' },
      { key: 'tutorialLoco', label: 'Ruta LOCO' },
      { key: 'tutorialDetection', label: 'Ruta Produccion / Deteccion' },
    ],
  },
]

function MaterialIcon({ name, className = '' }) {
  return (
    <span aria-hidden="true" className={`material-symbols-outlined ${className}`}>
      {name}
    </span>
  )
}

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
    projectTransfer: { group: 'configuration', tab: 'projectTransfer' },
    tutorialHub: { group: 'tutorial', tab: 'tutorialOverview' },
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
    'configuration:projectTransfer': 'projectTransfer',
    'tutorial:tutorialOverview': 'tutorialHub',
    'tutorial:tutorialScribble': 'tutorialHub',
    'tutorial:tutorialLoco': 'tutorialHub',
    'tutorial:tutorialDetection': 'tutorialHub',
  }
  return mapping[`${group}:${tab}`] || 'workbench'
}

export { GROUPS }

export default function Navigation({ activeGroup, onGroupChange, onTabChange, forceExpanded = false }) {
  const [collapsed, setCollapsed] = useState(true)
  const isCollapsed = forceExpanded ? false : collapsed

  function handleGroupClick(key) {
    const group = GROUPS.find((item) => item.key === key)
    const firstTab = group ? group.tabs[0].key : 'workbench'
    onGroupChange(key)
    onTabChange(firstTab)
  }

  return (
    <aside
      className={`sidebar ${isCollapsed ? 'collapsed' : 'expanded'}`}
      onMouseEnter={() => setCollapsed(false)}
      onMouseLeave={() => { if (!forceExpanded) setCollapsed(true) }}
    >
      <button
        className="sidebar-toggle"
        onClick={() => { if (!forceExpanded) setCollapsed((current) => !current) }}
        title={isCollapsed ? 'Expandir menu' : 'Colapsar menu'}
      >
        <MaterialIcon className="sidebar-toggle-icon" name={isCollapsed ? 'menu' : 'close'} />
        {!isCollapsed && <span className="sidebar-toggle-label">Menu</span>}
      </button>

      <div className="sidebar-section-label">
        {isCollapsed ? <MaterialIcon className="sidebar-section-icon" name="model_training" /> : 'Entrenamiento'}
      </div>
      {GROUPS.filter((group) => group.section === 'training').map((group) => (
        <button
          key={group.key}
          className={`sidebar-group-btn ${activeGroup === group.key ? 'active' : ''}`}
          onClick={() => handleGroupClick(group.key)}
          title={isCollapsed ? group.fullLabel : group.label}
          data-tour={`sidebar-group-${group.key}`}
        >
          <MaterialIcon className="sidebar-icon" name={group.icon} />
          {!isCollapsed && <span className="sidebar-label">{group.label}</span>}
        </button>
      ))}

      <div className="sidebar-separator" />
      <div className="sidebar-section-label">
        {isCollapsed ? <MaterialIcon className="sidebar-section-icon" name="precision_manufacturing" /> : 'Produccion'}
      </div>
      {GROUPS.filter((group) => group.section === 'production').map((group) => (
        <button
          key={group.key}
          className={`sidebar-group-btn ${activeGroup === group.key ? 'active' : ''}`}
          onClick={() => handleGroupClick(group.key)}
          title={isCollapsed ? group.fullLabel : group.label}
          data-tour={`sidebar-group-${group.key}`}
        >
          <MaterialIcon className="sidebar-icon" name={group.icon} />
          {!isCollapsed && <span className="sidebar-label">{group.label}</span>}
        </button>
      ))}

      <div className="sidebar-separator" />
      <div className="sidebar-section-label">
        {isCollapsed ? <MaterialIcon className="sidebar-section-icon" name="more_horiz" /> : 'Otros'}
      </div>
      {GROUPS.filter((group) => group.section === 'other').map((group) => (
        <button
          key={group.key}
          className={`sidebar-group-btn ${activeGroup === group.key ? 'active' : ''}`}
          onClick={() => handleGroupClick(group.key)}
          title={isCollapsed ? group.fullLabel : group.label}
          data-tour={`sidebar-group-${group.key}`}
        >
          <MaterialIcon className="sidebar-icon" name={group.icon} />
          {!isCollapsed && <span className="sidebar-label">{group.label}</span>}
        </button>
      ))}
    </aside>
  )
}
