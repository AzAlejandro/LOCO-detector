# GUI Navigation Redesign Plan

## Current State

The navigation currently renders as a horizontal bar above the main layout:

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ âœï¸ Grupo 1: Entrenamiento Scribble  â”‚ ðŸ”µ Grupo 2: ... â”‚ ðŸ“ Grupo 3: ... â”‚ â”€â”€ â”‚ ðŸ§ª Laboratorio LOCO [Experimental] â”‚
â”‚   Scribbles y Experimentos  â”‚  Revision de Resultados  â”‚  Modelos de Asistencia  â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚  â”Œâ”€â”€ left sidebar â”€â”€â”  â”‚  â”Œâ”€â”€ main content â”€â”€â”  â”‚
```

**Problems:**
- Group labels are very long ("Grupo 1: Entrenamiento Scribble") taking lots of horizontal space
- All 3 groups + LOCO Lab are shown simultaneously, creating visual noise
- Sub-tabs appear below groups as a second row, adding another layer
- The nav bar pushes the main content down

## Proposed Design (from recommendation)

Collapsible sidebar (Level 1) + horizontal tab strip above content (Level 2):

```
â”Œâ”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ âœï¸ â”‚ â”‚  Scribbles y Experimentos  â”‚  Revision de ...    â”‚  â† tab strip
â”‚    â”‚ â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚ ðŸ”µ â”‚ â”‚                                                   â”‚
â”‚    â”‚ â”‚              Main content area                    â”‚
â”‚ ðŸ“ â”‚ â”‚                                                   â”‚
â”‚    â”‚ â”‚                                                   â”‚
â”‚ ðŸ§ª â”‚ â”‚                                                   â”‚
â”‚ âš™ï¸ â”‚ â”‚                                                   â”‚
â””â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
  â†‘                          â†‘
sidebar (50px)           content area
collapsed                 with tab strip
```

When expanded (~200px):

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚ âœï¸ Scribbleâ”‚ â”‚  Scribbles y Experimentos  â”‚  Revision ... â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚ â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚
â”‚ ðŸ”µ LOCO   â”‚ â”‚                                            â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚ â”‚              Main content                   â”‚
â”‚ ðŸ“ Detecc.â”‚ â”‚                                            â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚ â”‚                                            â”‚
â”‚ ðŸ§ª Lab ðŸ…±ï¸ â”‚ â”‚                                            â”‚
â”‚â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚ â”‚                                            â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## Changes Required

### 1. [`frontend/src/components/Navigation.jsx`](frontend/src/components/Navigation.jsx)

**Rename groups** to short names:
| Current | New |
|---------|-----|
| `Grupo 1: Entrenamiento Scribble` | `Scribble` |
| `Grupo 2: Entrenamiento LOCO` | `LOCO` |
| `Grupo 3: Deteccion y Medicion` | `DetecciÃ³n` |
| `Laboratorio LOCO` | `Lab` |

**Restructure rendering:**
- Remove the horizontal `.nav-groups` bar and `.nav-subtabs` row
- Replace with a single `<aside>` sidebar element that:
  - Starts collapsed at `50px` width, showing only icons
  - Expands to `200px` on hover or toggle button click, showing icon + short label
  - Has a separator line between "Entrenamiento" group (Scribble + LOCO) and "ProducciÃ³n" (DetecciÃ³n)
  - Has another separator before "Lab" with a "Beta" badge
- Add a horizontal tab strip rendered ABOVE the main content area (not inside Navigation)

**Props changes:**
- Add `collapsed` boolean prop (or manage internally via state)
- Add `onToggleCollapse` callback

### 2. [`frontend/src/App.jsx`](frontend/src/App.jsx)

**Layout restructure:**
- Current: `<Navigation />` â†’ `<div className="layout"><aside className="left">...<section className="main">...`
- New: `<div className="app-layout"><aside className="sidebar">...<div className="content-area"><div className="tab-strip">...<div className="layout">...`

**Move tab strip to content area:**
- The Level 2 sub-tabs (Scribbles y Experimentos, Revision de Resultados, etc.) should render as a horizontal tab strip above the main content, not inside the Navigation component
- This tab strip only shows tabs for the currently active group

**Add sidebar collapse state:**
- New state: `sidebarCollapsed` (boolean, default `true`)
- Pass to Navigation component
- Toggle via button click or hover

### 3. [`frontend/src/styles.css`](frontend/src/styles.css)

**New CSS classes needed:**
- `.app-layout` â€” flex container for sidebar + content area
- `.sidebar` â€” the collapsible sidebar (width transitions between 50px and 200px)
- `.sidebar.collapsed` â€” 50px wide, only icons visible
- `.sidebar.expanded` â€” 200px wide, icons + labels visible
- `.sidebar-group-btn` â€” each group button in the sidebar
- `.sidebar-separator` â€” visual separator between training/production/lab sections
- `.sidebar-badge` â€” "Beta" badge for Lab
- `.tab-strip` â€” horizontal bar above main content
- `.tab-strip-btn` â€” individual tab buttons

**Remove (or keep for backward compat):**
- `.nav-container`, `.nav-groups`, `.nav-subtabs`, `.nav-group-btn`, `.nav-subtab-btn`, `.nav-separator`, `.nav-badge` â€” can be removed if Navigation is fully rewritten

### 4. [`frontend/src/components/Navigation.jsx`](frontend/src/components/Navigation.jsx) â€” Detailed rewrite

```jsx
const SIDEBAR_ITEMS = [
  { type: 'section', label: 'Entrenamiento' },
  { type: 'group', key: 'scribble', label: 'Scribble', icon: 'âœï¸' },
  { type: 'group', key: 'loco', label: 'LOCO', icon: 'ðŸ”µ' },
  { type: 'separator' },
  { type: 'section', label: 'ProducciÃ³n' },
  { type: 'group', key: 'detection', label: 'DetecciÃ³n', icon: 'ðŸ“' },
  { type: 'separator' },
  { type: 'group', key: 'locoLab', label: 'Lab', icon: 'ðŸ§ª', badge: 'Beta' },
]
```

The component renders the sidebar with these items. The tab strip (Level 2) is rendered separately by App.jsx using the `GROUPS` data.

## Implementation Steps

1. **Rewrite [`Navigation.jsx`](frontend/src/components/Navigation.jsx)**:
   - Change from horizontal bar to vertical sidebar
   - Add collapsed/expanded states
   - Rename group labels to short names
   - Add section separators (Entrenamiento / ProducciÃ³n)
   - Add Beta badge for Lab
   - Export `GROUPS` constant (already exported implicitly) for App.jsx to use for tab strip

2. **Update [`App.jsx`](frontend/src/App.jsx)**:
   - Add `sidebarCollapsed` state
   - Restructure layout: sidebar on left, content area on right
   - Add tab strip rendering above main content
   - The tab strip shows sub-tabs for the active group

3. **Update [`styles.css`](frontend/src/styles.css)**:
   - Add all new sidebar and tab strip CSS
   - Remove old nav CSS classes
   - Ensure responsive behavior (sidebar collapses fully on small screens)

## Mermaid Diagram

```mermaid
flowchart TD
    A[App.jsx] --> B[Navigation component]
    A --> C[Tab Strip - rendered in App.jsx]
    A --> D[Left sidebar - controls/params]
    A --> E[Main content - viewers/results]

    B --> F[Sidebar - Level 1 groups]
    F --> G[Scribble icon+label]
    F --> H[LOCO icon+label]
    F --> I[DetecciÃ³n icon+label]
    F --> J[Lab icon+label + Beta badge]

    C --> K[Sub-tabs for active group]
    K --> L[Scribbles y Experimentos]
    K --> M[Revision de Resultados]
    K --> N[Modelos de Asistencia]

    D --> O[Workbench / params panel]
    E --> P[Image viewer / curtain compare]
```

## Risks and Considerations

- **Backward compatibility**: The `legacyToGroup()` and `groupToLegacy()` functions should remain unchanged since they just map string keys
- **workspaceTab state**: Still used extensively in `useEffect` hooks for keyboard shortcuts and data loading â€” must keep it in sync
- **Responsive**: On screens < 1250px, the sidebar should collapse fully (0px width) and show a hamburger menu instead
- **Tab strip position**: Must be above the `.layout` grid but below the sidebar header â€” careful with CSS grid/flex nesting
- **No icon on sub-tabs**: The recommendation explicitly says sub-tabs should be text-only (no icons) to avoid visual noise
