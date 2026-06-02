import React from 'react'

function scopeLabel(scope) {
  if (scope === 'global') return 'General'
  if (scope === 'macro') return 'Macroseccion'
  if (scope === 'subtab') return 'Subpestaña'
  return scope
}

function statusLabel(progressEntry) {
  if (!progressEntry) return 'No iniciado'
  if (progressEntry.completed_at) return 'Completado'
  if (progressEntry.started_at) return 'En progreso'
  return 'No iniciado'
}

function formatTimestamp(value) {
  if (!value) return '-'
  const dt = new Date(value)
  if (Number.isNaN(dt.getTime())) return '-'
  return dt.toLocaleString()
}

function TutorialCard({
  tutorial,
  progressEntry,
  selected,
  onSelect,
  onStart,
  onStartFull,
  onReset,
}) {
  const isParent = tutorial.scope !== 'subtab' && Array.isArray(tutorial.children) && tutorial.children.length > 0
  const launches = Number(progressEntry?.launches || 0)
  return (
    <article
      className={`tutorial-card ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(tutorial.id)}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onSelect(tutorial.id)
        }
      }}
      role="button"
      tabIndex={0}
      data-tour={`tutorial-card-${tutorial.id}`}
    >
      <div className="tutorial-card-head">
        <div className="tutorial-card-titleblock">
          <strong>{tutorial.title}</strong>
          <span className="tutorial-chip">{scopeLabel(tutorial.scope)}</span>
        </div>
        <span className={`tutorial-status tutorial-status-${statusLabel(progressEntry).toLowerCase().replace(/\s+/g, '-')}`}>
          {statusLabel(progressEntry)}
        </span>
      </div>
      <p>{tutorial.description}</p>
      <div className="tutorial-card-meta">
        <span>{tutorial.estimatedSteps || tutorial.steps?.length || 0} pasos</span>
        <span>{launches} lanzamientos</span>
        {tutorial.workspaceTab ? <span>{tutorial.workspaceTab}</span> : null}
      </div>
      <div className="tutorial-card-actions">
        <button className="primary" type="button" onClick={(e) => { e.stopPropagation(); onStart(tutorial.id) }}>
          {progressEntry?.started_at && !progressEntry?.completed_at ? 'Continuar' : 'Iniciar'}
        </button>
        {isParent ? (
          <button type="button" onClick={(e) => { e.stopPropagation(); onStartFull(tutorial.id) }}>
            Recorrido completo
          </button>
        ) : null}
        <button type="button" onClick={(e) => { e.stopPropagation(); onReset(tutorial.id) }}>
          Reiniciar
        </button>
      </div>
    </article>
  )
}

export function TutorialSidebar({
  activeTutorialTab,
  selectedId,
  tutorialsByScope,
  progress,
  onSelect,
  onStart,
  onStartFull,
  onReset,
  tabConfig,
}) {
  const featuredTutorial = (tutorialsByScope.all || []).find((tutorial) => tutorial.id === tabConfig?.featured) || null
  const scopedSubtabs = tabConfig?.macroGroup
    ? (tutorialsByScope.subtab || []).filter((tutorial) => tutorial.macroGroup === tabConfig.macroGroup)
    : []
  return (
    <>
      <section className="card" data-tour="tutorial-hub-intro">
        <h2>{tabConfig?.title || 'Tutorial'}</h2>
        <p className="tutorial-copy">
          {tabConfig?.description || 'Centro de recorridos guiados.'}
        </p>
        {activeTutorialTab === 'tutorialOverview' ? (
          <div className="tutorial-plan-summary">
            <span>1 recorrido general</span>
            <span>3 recorridos por macroseccion</span>
            <span>{(tutorialsByScope.subtab || []).length} recorridos por subpestana</span>
          </div>
        ) : null}
      </section>

      {featuredTutorial ? (
        <section className="card tutorial-list-card">
          <h2>{activeTutorialTab === 'tutorialOverview' ? 'Recorrido general' : featuredTutorial.title}</h2>
          <div className="tutorial-list">
            <TutorialCard
              tutorial={featuredTutorial}
              progressEntry={progress[featuredTutorial.id]}
              selected={selectedId === featuredTutorial.id}
              onSelect={onSelect}
              onStart={onStart}
              onStartFull={onStartFull}
              onReset={onReset}
            />
          </div>
        </section>
      ) : null}

      {scopedSubtabs.length ? (
        <section className="card tutorial-list-card">
          <div className="tutorial-subsection-head">
            <div>
              <h2>Subtutoriales</h2>
              <p className="tutorial-copy">
                Recorridos detallados de las subpestañas incluidas en esta ruta.
              </p>
            </div>
          </div>
          <div className="tutorial-list">
            {scopedSubtabs.map((tutorial) => (
              <TutorialCard
                key={tutorial.id}
                tutorial={tutorial}
                progressEntry={progress[tutorial.id]}
                selected={selectedId === tutorial.id}
                onSelect={onSelect}
                onStart={onStart}
                onStartFull={onStartFull}
                onReset={onReset}
              />
            ))}
          </div>
        </section>
      ) : null}
    </>
  )
}

export function TutorialViewer({
  tutorial,
  progressEntry,
  onStart,
  onStartFull,
  onReset,
  onResetAll,
}) {
  if (!tutorial) {
    return (
      <section className="tutorial-detail-empty" data-tour="workspace-panel-tutorialHub">
        <h2>Tutoriales</h2>
        <p>Selecciona un tutorial para ver su detalle.</p>
      </section>
    )
  }
  const isParent = tutorial.scope !== 'subtab' && Array.isArray(tutorial.children) && tutorial.children.length > 0
  return (
    <section className="tutorial-detail" data-tour="workspace-panel-tutorialHub">
      <div className="tutorial-detail-head">
        <div>
          <span className="tutorial-scope">{scopeLabel(tutorial.scope)}</span>
          <h2>{tutorial.title}</h2>
          <p>{tutorial.description}</p>
        </div>
        <div className="tutorial-detail-actions">
          <button className="primary" onClick={() => onStart(tutorial.id)}>
            {progressEntry?.started_at && !progressEntry?.completed_at ? 'Continuar' : 'Iniciar'}
          </button>
          {isParent ? <button onClick={() => onStartFull(tutorial.id)}>Recorrido completo</button> : null}
          <button onClick={() => onReset(tutorial.id)}>Reiniciar</button>
          <button className="bad" onClick={onResetAll}>Limpiar progreso</button>
        </div>
      </div>

      <div className="tutorial-detail-grid">
        <div className="tutorial-detail-card">
          <h3>Objetivo</h3>
          <p>{tutorial.purpose}</p>
        </div>
        <div className="tutorial-detail-card">
          <h3>Estado</h3>
          <p>{statusLabel(progressEntry)}</p>
          <p className="tutorial-muted">Ultimo inicio: {formatTimestamp(progressEntry?.started_at)}</p>
          <p className="tutorial-muted">Ultima finalizacion: {formatTimestamp(progressEntry?.completed_at)}</p>
        </div>
      </div>

      <div className="tutorial-detail-panels">
        <div className="tutorial-detail-card">
          <h3>Al terminar el usuario debe entender</h3>
          <ul>
            {(tutorial.understands || []).map((item) => <li key={item}>{item}</li>)}
          </ul>
        </div>
        <div className="tutorial-detail-card">
          <h3>Pasos base del tour</h3>
          <ol>
            {(tutorial.steps || []).map((step, idx) => (
              <li key={`${tutorial.id}-step-${idx}`}>
                <strong>{step.title}</strong>
                <span>{step.description}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  )
}
