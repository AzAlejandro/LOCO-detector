import React from 'react'

/**
 * Navigation component for hierarchical workspace tabs.
 *
 * Level 1: 3 logical groups + LOCO Lab as separate experimental tab
 * Level 2: Sub-tabs within each group
 *
 * Props:
 *   activeGroup: 'scribble' | 'loco' | 'detection' | 'locoLab'
 *   activeTab: string (sub-tab within the active group)
 *   onGroupChange: (group) => void
 *   onTabChange: (tab) => void
 */

const GROUPS = [
  {
    key: 'scribble',
    label: 'Grupo 1: Entrenamiento Scribble',
    icon: '✏️',
    tabs: [
      { key: 'workbench', label: 'Scribbles y Experimentos' },
      { key: 'review', label: 'Revision de Resultados' },
      { key: 'models', label: 'Modelos de Asistencia' },
    ],
  },
  {
    key: 'loco',
    label: 'Grupo 2: Entrenamiento LOCO',
    icon: '🔵',
    tabs: [
      { key: 'locoDataset', label: 'Generar Dataset' },
      { key: 'locoAugment', label: 'Aumentacion' },
      { key: 'locoTraining', label: 'Entrenamiento' },
      { key: 'locoTest', label: 'Test de Modelo' },
    ],
  },
  {
    key: 'detection',
    label: 'Grupo 3: Deteccion y Medicion',
    icon: '📏',
    tabs: [
      { key: 'locoModel', label: 'Detector LOCO' },
      { key: 'diameter', label: 'Medicion de Diametros' },
    ],
  },
]

const LOCO_LAB_TAB = { key: 'loco', label: 'Laboratorio LOCO', icon: '🧪' }

/**
 * Map legacy workspaceTab to {activeGroup, activeTab}.
 */
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
    loco: { group: 'locoLab', tab: 'loco' },
  }
  return mapping[tab] || { group: 'scribble', tab: 'workbench' }
}

/**
 * Map {activeGroup, activeTab} back to legacy workspaceTab.
 */
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
    'locoLab:loco': 'loco',
  }
  return mapping[`${group}:${tab}`] || 'workbench'
}

export default function Navigation({ activeGroup, activeTab, onGroupChange, onTabChange }) {
  const isLocoLab = activeGroup === 'locoLab'

  return (
    <div className="nav-container">
      {/* Level 1: Group tabs */}
      <div className="nav-groups">
        {GROUPS.map((g) => (
          <button
            key={g.key}
            className={`nav-group-btn ${activeGroup === g.key && !isLocoLab ? 'active' : ''}`}
            onClick={() => {
              // When switching to a group, keep the current sub-tab if it belongs to that group,
              // otherwise select the first tab of the group
              const groupTabs = GROUPS.find((x) => x.key === g.key)
              const firstTab = groupTabs ? groupTabs.tabs[0].key : g.tabs[0].key
              onGroupChange(g.key)
              onTabChange(firstTab)
            }}
            title={g.label}
          >
            <span className="nav-icon">{g.icon}</span>
            <span className="nav-label">{g.label}</span>
          </button>
        ))}
        {/* LOCO Lab as separate experimental tab */}
        <div className="nav-separator" />
        <button
          key={LOCO_LAB_TAB.key}
          className={`nav-group-btn nav-lab-btn ${isLocoLab ? 'active' : ''}`}
          onClick={() => {
            onGroupChange('locoLab')
            onTabChange('loco')
          }}
          title={LOCO_LAB_TAB.label}
        >
          <span className="nav-icon">{LOCO_LAB_TAB.icon}</span>
          <span className="nav-label">{LOCO_LAB_TAB.label}</span>
          <span className="nav-badge">Experimental</span>
        </button>
      </div>

      {/* Level 2: Sub-tabs (only for non-locoLab groups) */}
      {!isLocoLab && (
        <div className="nav-subtabs">
          {GROUPS.filter((g) => g.key === activeGroup).map((g) =>
            g.tabs.map((t) => (
              <button
                key={t.key}
                className={`nav-subtab-btn ${activeTab === t.key ? 'active' : ''}`}
                onClick={() => onTabChange(t.key)}
              >
                {t.label}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  )
}
