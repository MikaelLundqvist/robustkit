import numpy as np
import pandas as pd
import pytest

from robustkit import (
    conditional_mutual_information,
    communication_score, rank_by_communication,
    pair_redundancy, pair_synergy, rank_communicative_pairs,
)


def make_interaction_df(n=800, seed=0):
    """
    A dataset with a genuine interaction effect: `region` only matters
    for the target *within* certain values of `segment` -- i.e.
    segment should show synergy with region, not just marginal
    relevance on its own.

    Also includes a fully redundant duplicate of `segment` (`segment_copy`)
    and an irrelevant, high-cardinality noise column.
    """
    rng = np.random.default_rng(seed)
    segment = rng.choice(["A", "B"], size=n)
    region = rng.choice(["North", "South"], size=n)
    segment_copy = segment.copy()  # perfectly redundant with segment
    noise = rng.choice([f"n{i}" for i in range(15)], size=n)

    # region only affects the target when segment == "A"
    region_effect = np.where((segment == "A") & (region == "North"), 20.0, 0.0)
    target = 50 + 10 * (segment == "A") + region_effect + rng.normal(0, 2, n)

    return pd.DataFrame({
        "segment": segment,
        "region": region,
        "segment_copy": segment_copy,
        "noise": noise,
        "target": target,
    })


def test_conditional_mi_is_nonnegative_and_finite():
    df = make_interaction_df()
    cmi = conditional_mutual_information(df, feature="region", target="target", condition_on="segment")
    assert cmi >= 0
    assert np.isfinite(cmi)


def test_conditional_mi_captures_interaction_effect():
    """
    region's marginal relationship with target should be weak (the
    effect only exists within segment == "A"), but conditioning on
    segment should reveal a much stronger relationship.
    """
    from robustkit.information.mutual_info import _mutual_info_between
    df = make_interaction_df()

    marginal = _mutual_info_between(df["region"], df["target"])
    conditional = conditional_mutual_information(df, "region", "target", condition_on="segment")

    assert conditional > marginal


def test_communication_score_returns_expected_keys():
    df = make_interaction_df()
    result = communication_score(df, feature="segment", target="target")
    expected_keys = {
        "feature", "raw_mutual_information", "mutual_information", "stability",
        "group_size_score", "compressibility", "interpretability", "communication_index",
    }
    assert expected_keys <= set(result.keys())
    assert 0 <= result["communication_index"] <= 1


def test_communication_score_penalizes_small_groups():
    df = make_interaction_df(n=100)
    # Force a feature with a tiny minority group
    df["rare_flag"] = ["rare"] * 3 + ["common"] * 97

    result = communication_score(df, feature="rare_flag", target="target", min_group_size=20)
    assert result["group_size_score"] < 0.2  # smallest group (3) is far below min_group_size (20)


def test_rank_by_communication_orders_and_covers_all_features():
    df = make_interaction_df()
    ranking = rank_by_communication(df, target="target")
    assert set(ranking["feature"]) == {"segment", "region", "segment_copy", "noise"}
    assert (ranking["communication_index"].diff().dropna() <= 1e-9).all()  # sorted descending


def test_pair_redundancy_detects_duplicate_feature():
    df = make_interaction_df()
    redundancy_duplicate = pair_redundancy(df, "segment", "segment_copy")
    redundancy_unrelated = pair_redundancy(df, "segment", "noise")

    assert redundancy_duplicate == pytest.approx(1.0, abs=0.05)
    assert redundancy_duplicate > redundancy_unrelated


def test_pair_synergy_detects_interaction_effect():
    df = make_interaction_df()
    synergy = pair_synergy(df, feature_1="segment", feature_2="region", target="target")
    assert synergy > 0  # region's info about target increases once segment is known


def test_rank_communicative_pairs_ranks_interacting_pair_above_redundant_pair():
    df = make_interaction_df()
    pairs = rank_communicative_pairs(df, target="target")

    assert len(pairs) == 6  # C(4, 2) candidate pairs

    def _get_pair_row(pairs_df, a, b):
        mask = (
            ((pairs_df["feature_1"] == a) & (pairs_df["feature_2"] == b))
            | ((pairs_df["feature_1"] == b) & (pairs_df["feature_2"] == a))
        )
        return pairs_df[mask].iloc[0]

    interacting_pair = _get_pair_row(pairs, "segment", "region")
    redundant_pair = _get_pair_row(pairs, "segment", "segment_copy")

    assert interacting_pair["pair_score"] > redundant_pair["pair_score"]
    assert redundant_pair["redundancy"] > interacting_pair["redundancy"]
