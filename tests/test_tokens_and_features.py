"""Tokenisation ladder and the multi-channel journey representation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from journey_clustering import tokens as T
from journey_clustering.config import CanonizeConfig, FeatureConfig, TokenConfig
from journey_clustering.features import JourneyVectorizer
from journey_clustering.preprocessing import canonicalize_frame


def canonical_fixture() -> pd.DataFrame:
    """A short, realistic modem-control journey on both platforms."""
    rows = [
        ("iOS", "view", "ManageModemVC", "ManageModemVC"),
        ("iOS", "action", "do_action/MODEM_RESET", None),
        ("iOS", "action", "btn_back", None),
        ("Android", "view", "android/home/home_service_management/internet_service_tab/modem_control",
         "android/home/home_service_management/internet_service_tab/modem_control"),
        ("Android", "action", "android/home/home_service_management/internet_service_tab/modem_control/reboot_modem", None),
        ("Android", "action", "btn_back", None),
    ]
    frame = pd.DataFrame(
        [
            {
                "session_id": f"s-{platform}",
                "platform": platform,
                "key": key,
                "segmentation_name": name,
                "screen_id": screen,
                "client_time": 1775001600000 + index * 1000,
            }
            for index, (platform, key, name, screen) in enumerate(rows)
        ]
    )
    canonical, _ = canonicalize_frame(frame, source_file="fixture.csv")
    return canonical


class TokenLadderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = T.build_tokens(canonical_fixture(), CanonizeConfig())

    def test_levels_are_real_resolutions_not_aliases(self) -> None:
        distinct = {column: self.tokens[column].nunique() for column in T.TOKEN_COLUMNS}
        self.assertGreater(distinct["exact_token"], distinct["token_l3"])
        self.assertGreater(distinct["token_l3"], distinct["token_l2"])
        self.assertGreaterEqual(distinct["token_l2"], distinct["token_l1"])
        self.assertEqual(distinct["token_l1"], 1)  # every row is `internet`

    def test_level_shape(self) -> None:
        row = self.tokens.iloc[0]
        self.assertEqual(row["token_l1"], "internet")
        self.assertEqual(row["token_l2"], "internet/modem")
        self.assertEqual(row["token_l3"], "internet/modem/modem/view")
        self.assertEqual(row["operation_token"], "entry:view")

    def test_the_two_platforms_share_every_semantic_token(self) -> None:
        by_platform = self.tokens.groupby("platform")
        android = by_platform.get_group("android")
        ios = by_platform.get_group("ios")
        # exact tokens are disjoint by construction; semantic ones must not be
        self.assertFalse(set(android["exact_token"]) & set(ios["exact_token"]))
        self.assertEqual(set(android["token_l2"]), set(ios["token_l2"]))
        self.assertEqual(set(android["operation_token"]), set(ios["operation_token"]))

    def test_source_event_columns_are_unchanged(self) -> None:
        source = canonical_fixture()
        pd.testing.assert_frame_equal(self.tokens[source.columns.tolist()], source)

    def test_unknown_level_and_channel_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown token level"):
            T.level_column("L4")
        with self.assertRaisesRegex(ValueError, "unknown token channel"):
            T.channel_column("vibes")

    def test_channels_can_be_rebuilt_from_exact_sequences_alone(self) -> None:
        exact = [self.tokens["exact_token"].tolist()]
        rebuilt = T.channels_from_exact_sequences(exact, ("coarse", "intent", "operation"))
        self.assertEqual(rebuilt["coarse"][0], self.tokens["token_l2"].tolist())
        self.assertEqual(rebuilt["intent"][0], self.tokens["token_l3"].tolist())
        self.assertEqual(rebuilt["operation"][0], self.tokens["operation_token"].tolist())

    def test_rebuilding_leaves_already_coarse_and_rare_tokens_alone(self) -> None:
        rebuilt = T.channels_from_exact_sequences([[T.RARE, "internet/modem"]], ("coarse",))
        self.assertEqual(rebuilt["coarse"][0], [T.RARE, "internet/modem"])


class RareFoldingTest(unittest.TestCase):
    def test_a_rare_token_backs_off_to_its_coarse_form(self) -> None:
        sequences = [["a", "rare_once"], ["a", "b"], ["a", "b"]]
        coarse = [["A", "COARSE"], ["A", "B"], ["A", "B"]]
        folded, stats = T.fold_rare_tokens(sequences, coarse, TokenConfig(min_journey_df=3))
        # neither `rare_once` nor its coarse form clears the journey threshold
        self.assertEqual(folded[0], ["a", T.RARE])
        self.assertEqual(int(stats.loc[0, "tokens_kept"]), 3)
        self.assertEqual(int(stats.loc[0, "tokens_backed_off"]), 0)

        shared_coarse = [["A", "A"], ["A", "A"], ["A", "A"]]
        folded, stats = T.fold_rare_tokens(sequences, shared_coarse, TokenConfig(min_journey_df=3))
        self.assertEqual(folded[0], ["a", "A"])
        self.assertEqual(int(stats.loc[0, "tokens_backed_off"]), 3)
        self.assertEqual(int(stats.loc[0, "tokens_to_rare"]), 0)


def journey_frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_events_final": np.arange(4, 4 + n, dtype=float),
            "n_unique_tokens": np.full(n, 3.0),
            "action_ratio": np.linspace(0.1, 0.9, n),
            "back_rate": np.linspace(0.0, 0.5, n),
            "revisit_ratio": np.linspace(0.0, 0.3, n),
            "n_loop_removed": np.zeros(n),
            "n_dedup_removed": np.zeros(n),
            "span_seconds": np.linspace(10, 100, n),
            "median_gap_s": np.full(n, 2.0),
            "p90_gap_s": np.full(n, 5.0),
            "max_gap_s": np.full(n, 9.0),
        }
    )


class MultiChannelVectorizerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n = 12
        self.journeys = journey_frame(self.n)
        self.sequences = [
            ["view@ManageModemVC", "action@ManageModemVC#do_action/MODEM_RESET"]
            if index % 2
            else ["view@PaymentHomeVC", "action@PaymentHomeVC#home/payment/pay"]
            for index in range(self.n)
        ]
        self.channels = T.channels_from_exact_sequences(
            self.sequences, ("coarse", "intent", "operation")
        )

    def test_channels_widen_the_representation(self) -> None:
        single = JourneyVectorizer(FeatureConfig(min_df=1, svd_components=4), channel_weights={})
        base, base_info = single.fit_transform(self.journeys, self.sequences)

        multi = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4),
            channel_weights={"coarse": 0.5, "intent": 0.3, "operation": 0.2},
        )
        wide, wide_info = multi.fit_transform(self.journeys, self.sequences, self.channels)

        self.assertEqual(base.shape[0], wide.shape[0])
        self.assertGreater(wide.shape[1], base.shape[1])
        self.assertEqual(sorted(wide_info["channels"]), ["coarse", "intent", "operation", "primary"])
        self.assertNotIn("channels", base_info.get("channels", {}))

    def coarse_only(self) -> dict[str, list[list[str]]]:
        return {"coarse": self.channels["coarse"]}

    def test_transform_reproduces_fit_transform(self) -> None:
        vectorizer = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4), channel_weights={"coarse": 0.5}
        )
        fitted, _ = vectorizer.fit_transform(self.journeys, self.sequences, self.coarse_only())
        again = vectorizer.transform(self.journeys, self.sequences, self.coarse_only())
        np.testing.assert_allclose(fitted, again, rtol=1e-9, atol=1e-9)

    def test_global_pca_reduces_weighted_matrix_and_round_trips(self) -> None:
        vectorizer = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4, global_pca_components=3),
            channel_weights={"coarse": 0.5},
        )
        fitted, info = vectorizer.fit_transform(self.journeys, self.sequences, self.coarse_only())
        again = vectorizer.transform(self.journeys, self.sequences, self.coarse_only())
        self.assertEqual(fitted.shape, (self.n, 3))
        self.assertEqual(info["global_pca_components"], 3)
        np.testing.assert_allclose(fitted, again, rtol=1e-8, atol=1e-8)

    def test_channel_weight_scales_only_its_own_block(self) -> None:
        light = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4), channel_weights={"coarse": 0.1}
        )
        heavy = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4), channel_weights={"coarse": 1.0}
        )
        light_matrix, _ = light.fit_transform(self.journeys, self.sequences, self.coarse_only())
        heavy_matrix, _ = heavy.fit_transform(self.journeys, self.sequences, self.coarse_only())
        self.assertEqual(light_matrix.shape, heavy_matrix.shape)
        self.assertLess(np.abs(light_matrix).sum(), np.abs(heavy_matrix).sum())

    def test_a_missing_or_unexpected_channel_is_an_error_not_a_silent_zero(self) -> None:
        vectorizer = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4), channel_weights={"coarse": 0.5}
        )
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            vectorizer.fit_transform(self.journeys, self.sequences)
        vectorizer.fit_transform(self.journeys, self.sequences, self.coarse_only())
        with self.assertRaisesRegex(ValueError, "channel mismatch"):
            vectorizer.transform(self.journeys, self.sequences, {"intent": self.channels["intent"]})

    def test_zero_weight_channels_are_dropped(self) -> None:
        vectorizer = JourneyVectorizer(
            FeatureConfig(min_df=1, svd_components=4),
            channel_weights={"coarse": 0.5, "intent": 0.0},
        )
        self.assertEqual(sorted(vectorizer.channel_weights), ["coarse"])


if __name__ == "__main__":
    unittest.main()
