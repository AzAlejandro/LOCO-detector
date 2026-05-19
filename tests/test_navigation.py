"""Tests for the hierarchical navigation mapping logic.

These tests validate the legacyToGroup and groupToLegacy mapping functions
that bridge the old flat workspaceTab (0-8) with the new activeGroup/activeTab
hierarchical navigation. No DOM/browser required.
"""

from __future__ import annotations

# ── Inline copy of the Navigation mapping logic ──────────────────────────────

GROUPS = [
    {
        'key': 'group1',
        'label': 'Grupo 1',
        'tabs': [
            {'key': 'scribbles', 'label': 'Scribbles'},
            {'key': 'experiments', 'label': 'Experimentos'},
            {'key': 'model', 'label': 'Modelo'},
        ],
    },
    {
        'key': 'group2',
        'label': 'Grupo 2',
        'tabs': [
            {'key': 'dataset', 'label': 'Dataset'},
            {'key': 'training', 'label': 'Entrenamiento'},
            {'key': 'loco-model', 'label': 'LOCO Model'},
        ],
    },
    {
        'key': 'group3',
        'label': 'Grupo 3',
        'tabs': [
            {'key': 'detector', 'label': 'Detector'},
            {'key': 'diameter', 'label': 'Diameter'},
            {'key': 'review', 'label': 'Review'},
        ],
    },
]

# Legacy flat tab order (0-8)
LEGACY_ORDER = ['scribbles', 'experiments', 'model', 'dataset', 'training', 'loco-model', 'detector', 'diameter', 'review']


def legacy_to_group(legacy_tab: str) -> tuple[str, str]:
    """Map a legacy flat tab key to (group_key, tab_key)."""
    for group in GROUPS:
        for tab in group['tabs']:
            if tab['key'] == legacy_tab:
                return group['key'], tab['key']
    return 'group1', 'scribbles'


def group_to_legacy(group_key: str, tab_key: str) -> str:
    """Map (group_key, tab_key) back to the legacy flat tab key."""
    for group in GROUPS:
        if group['key'] == group_key:
            for tab in group['tabs']:
                if tab['key'] == tab_key:
                    return tab['key']
    return 'scribbles'


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLegacyToGroup:
    def test_all_tabs_map_correctly(self) -> None:
        """Every legacy tab maps to the correct group."""
        assert legacy_to_group('scribbles') == ('group1', 'scribbles')
        assert legacy_to_group('experiments') == ('group1', 'experiments')
        assert legacy_to_group('model') == ('group1', 'model')
        assert legacy_to_group('dataset') == ('group2', 'dataset')
        assert legacy_to_group('training') == ('group2', 'training')
        assert legacy_to_group('loco-model') == ('group2', 'loco-model')
        assert legacy_to_group('detector') == ('group3', 'detector')
        assert legacy_to_group('diameter') == ('group3', 'diameter')
        assert legacy_to_group('review') == ('group3', 'review')

    def test_unknown_tab_falls_back(self) -> None:
        assert legacy_to_group('unknown') == ('group1', 'scribbles')
        assert legacy_to_group('') == ('group1', 'scribbles')

    def test_loco_tab_not_in_groups(self) -> None:
        """'loco' tab is handled separately (LOCO Lab experimental)."""
        result = legacy_to_group('loco')
        assert result == ('group1', 'scribbles'), 'loco should fall back to default'


class TestGroupToLegacy:
    def test_all_groups_map_correctly(self) -> None:
        assert group_to_legacy('group1', 'scribbles') == 'scribbles'
        assert group_to_legacy('group1', 'experiments') == 'experiments'
        assert group_to_legacy('group1', 'model') == 'model'
        assert group_to_legacy('group2', 'dataset') == 'dataset'
        assert group_to_legacy('group2', 'training') == 'training'
        assert group_to_legacy('group2', 'loco-model') == 'loco-model'
        assert group_to_legacy('group3', 'detector') == 'detector'
        assert group_to_legacy('group3', 'diameter') == 'diameter'
        assert group_to_legacy('group3', 'review') == 'review'

    def test_unknown_group_falls_back(self) -> None:
        assert group_to_legacy('group_unknown', 'scribbles') == 'scribbles'
        assert group_to_legacy('group1', 'unknown_tab') == 'scribbles'

    def test_round_trip(self) -> None:
        """legacy_to_group then group_to_legacy returns the original key."""
        for tab in LEGACY_ORDER:
            group_key, tab_key = legacy_to_group(tab)
            restored = group_to_legacy(group_key, tab_key)
            assert restored == tab, f'Round-trip failed for {tab}: got {restored}'


class TestGroupStructure:
    def test_all_legacy_tabs_covered(self) -> None:
        """Every legacy tab appears in exactly one group."""
        found: set[str] = set()
        for group in GROUPS:
            for tab in group['tabs']:
                found.add(tab['key'])
        for tab in LEGACY_ORDER:
            assert tab in found, f'Legacy tab {tab} not found in any group'

    def test_no_duplicate_tabs(self) -> None:
        """No tab key appears in more than one group."""
        all_tabs: list[str] = []
        for group in GROUPS:
            for tab in group['tabs']:
                all_tabs.append(tab['key'])
        assert len(all_tabs) == len(set(all_tabs)), 'Duplicate tab keys found'

    def test_three_groups_with_three_tabs_each(self) -> None:
        assert len(GROUPS) == 3
        for group in GROUPS:
            assert len(group['tabs']) == 3, f'Group {group["key"]} does not have 3 tabs'

    def test_loco_tab_is_separate(self) -> None:
        """'loco' should NOT be in any group (it's the experimental LOCO Lab tab)."""
        all_tabs: list[str] = []
        for group in GROUPS:
            for tab in group['tabs']:
                all_tabs.append(tab['key'])
        assert 'loco' not in all_tabs, 'loco tab should not be in any group'
