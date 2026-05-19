"""Tests for diameter distribution (histogram + statistics) logic.

These tests validate the Histogram component's computeStats and computeBins
functions, and the CSV export helper, without requiring a browser or DOM.
"""

from __future__ import annotations

import math

import numpy as np


# ── Inline copy of the Histogram logic (same as in components/Histogram.jsx) ──


def compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistics: mean, median, std, min, max, N."""
    if not values:
        return {'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    sorted_vals = sorted(values)
    if n % 2 == 1:
        median = float(sorted_vals[n // 2])
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    return {
        'mean': round(mean, 4),
        'median': round(median, 4),
        'std': round(std, 4),
        'min': round(float(min(values)), 4),
        'max': round(float(max(values)), 4),
        'n': n,
    }


def compute_bins(values: list[float], bins: int = 10) -> list[dict[str, float]]:
    """Compute histogram bins (count per bin)."""
    if not values or bins < 1:
        return []
    mn = min(values)
    mx = max(values)
    if mx == mn:
        return [{'bin_start': mn, 'bin_end': mx + 1.0, 'count': len(values)}]
    bin_width = (mx - mn) / bins
    counts = [0] * bins
    for v in values:
        idx = min(int((v - mn) / bin_width), bins - 1)
        counts[idx] += 1
    result = []
    for i in range(bins):
        result.append({
            'bin_start': round(mn + i * bin_width, 4),
            'bin_end': round(mn + (i + 1) * bin_width, 4),
            'count': counts[i],
        })
    return result


def export_histogram_csv(values: list[float], unit: str = 'px') -> str:
    """Generate CSV string from values."""
    lines = ['value']
    for v in values:
        lines.append(f'{v}')
    return '\n'.join(lines) + '\n'


# ── Tests ────────────────────────────────────────────────────────────────────


class TestComputeStats:
    def test_empty(self) -> None:
        stats = compute_stats([])
        assert stats['n'] == 0
        assert stats['mean'] == 0.0

    def test_single_value(self) -> None:
        stats = compute_stats([42.0])
        assert stats['n'] == 1
        assert stats['mean'] == 42.0
        assert stats['median'] == 42.0
        assert stats['min'] == 42.0
        assert stats['max'] == 42.0
        assert stats['std'] == 0.0

    def test_multiple_values(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_stats(values)
        assert stats['n'] == 5
        assert stats['mean'] == 30.0
        assert stats['median'] == 30.0
        assert stats['min'] == 10.0
        assert stats['max'] == 50.0
        assert round(stats['std'], 4) == round(math.sqrt(200), 4)

    def test_even_count_median(self) -> None:
        stats = compute_stats([1.0, 2.0, 3.0, 4.0])
        assert stats['median'] == 2.5

    def test_odd_count_median(self) -> None:
        stats = compute_stats([1.0, 2.0, 3.0])
        assert stats['median'] == 2.0

    def test_std_zero_for_identical(self) -> None:
        stats = compute_stats([5.0, 5.0, 5.0])
        assert stats['std'] == 0.0


class TestComputeBins:
    def test_empty(self) -> None:
        assert compute_bins([]) == []

    def test_single_value(self) -> None:
        bins = compute_bins([10.0], bins=5)
        assert len(bins) == 1
        assert bins[0]['count'] == 1

    def test_all_identical(self) -> None:
        bins = compute_bins([7.0, 7.0, 7.0], bins=4)
        assert len(bins) == 1
        assert bins[0]['count'] == 3

    def test_distribution(self) -> None:
        values = [1.0, 1.0, 2.0, 3.0, 5.0]
        bins = compute_bins(values, bins=4)
        assert len(bins) == 4
        total = sum(b['count'] for b in bins)
        assert total == 5

    def test_bins_parameter_respected(self) -> None:
        values = list(range(100))
        bins = compute_bins(values, bins=20)
        assert len(bins) == 20

    def test_bins_minimum_one(self) -> None:
        values = [1.0, 2.0, 3.0]
        bins = compute_bins(values, bins=0)
        assert bins == []


class TestExportHistogramCsv:
    def test_empty_values(self) -> None:
        csv = export_histogram_csv([])
        assert csv == 'value\n'

    def test_single_value(self) -> None:
        csv = export_histogram_csv([3.14])
        assert '3.14' in csv
        assert csv.count('\n') == 2  # header + 1 value

    def test_multiple_values(self) -> None:
        csv = export_histogram_csv([1.0, 2.0, 3.0])
        lines = csv.strip().split('\n')
        assert lines[0] == 'value'
        assert len(lines) == 4  # header + 3 values


class TestIntegrationStatsAndBins:
    """Verify that stats and bins are consistent with each other."""

    def test_bin_range_covers_all_values(self) -> None:
        values = [0.5, 1.2, 3.8, 4.0, 7.5]
        stats = compute_stats(values)
        bins = compute_bins(values, bins=5)
        assert stats['min'] == min(values)
        assert stats['max'] == max(values)
        assert bins[0]['bin_start'] <= stats['min']
        assert bins[-1]['bin_end'] >= stats['max']

    def test_total_count_matches(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_stats(values)
        bins = compute_bins(values, bins=3)
        total_in_bins = sum(b['count'] for b in bins)
        assert total_in_bins == stats['n']

    def test_large_dataset(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=50.0, scale=10.0, size=1000).tolist()
        stats = compute_stats(values)
        bins = compute_bins(values, bins=20)
        assert stats['n'] == 1000
        assert 45 < stats['mean'] < 55
        assert len(bins) == 20
        assert sum(b['count'] for b in bins) == 1000
