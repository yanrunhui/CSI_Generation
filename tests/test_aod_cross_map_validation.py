from __future__ import annotations

from scripts.aggregate_aod_cross_map_validation import (
    _map_cluster_summary,
    _seed_stability,
)


def test_map_cluster_summary_weights_maps_not_rows() -> None:
    rows = [
        {"map_id": 1, "value": 0.0},
        {"map_id": 1, "value": 2.0},
        {"map_id": 2, "value": 4.0},
    ]
    result = _map_cluster_summary(
        rows,
        lambda row: row["value"],
        bootstrap_samples=200,
        seed=7,
    )
    assert result["maps"] == 2
    assert result["observations"] == 3
    assert result["map_cluster_mean"] == 2.5


def test_seed_stability_requires_distinct_b3_cache_seeds() -> None:
    common = {
        "map_id": 10,
        "L": 5,
        "los_status": 0,
        "k_factor_valid": False,
        "mode": "mean",
        "target_az_deg": 90.0,
        "tolerance_deg": 10.0,
        "natural_acceptance_rate": 0.1,
        "status": "insufficient_fidelity_support",
    }
    same_seed = [
        {**common, "report_label": "a", "b3_cache_generation_seed": 2026},
        {**common, "report_label": "b", "b3_cache_generation_seed": 2026},
    ]
    assert (
        _seed_stability(same_seed)["groups_with_multiple_generation_seeds"] == 0
    )
    different_seed = [
        same_seed[0],
        {**same_seed[1], "b3_cache_generation_seed": 2027},
    ]
    assert (
        _seed_stability(different_seed)["groups_with_multiple_generation_seeds"]
        == 1
    )
