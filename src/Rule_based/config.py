"""Central configuration for the journey pipeline.

Every tunable lives here so that a run is fully described by one object and
sweeps stay reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------
# Screen taxonomy
# --------------------------------------------------------------------------
# Container / OS-chrome view controllers. These are emitted by the navigation
# framework rather than chosen by the user. Kept as a *label*, never silently
# merged with anything else.
CHROME_SCREENS: frozenset[str] = frozenset(
    {
        # iOS navigation + hosting containers
        "MainTabBarController",
        "BaseNavigation",
        "UINavigationController",
        "UIViewController",
        "UITrackingElementWindowController",
        "_UISceneHostingViewController",
        "UIHostingController<PopupView>",
        "SFBrowserRemoteViewController",
        "SFAuthenticationViewController",
        "SFSafariViewController",
        # webview shells: the shell itself carries no destination information,
        # the URL that follows it does
        "WebkitEcommerceController",
        "HiWebViewActivity",
        "WebViewActivity",
    } 
)
# App-launch screens. Useful as journey anchors, useless as journey content.
BOOT_SCREENS: frozenset[str] = frozenset(
    {"SplashVC", "SplashActivity", "Splash", "MainAppActivity", "LaunchScreen"}
)

# Root / hub screens. Arriving here usually means "previous goal finished".
#
# NOTE: MainTabBarController / MainAppActivity are deliberately EXCLUDED even
# though they are the literal tab-bar root. iOS pushes through
# MainTabBarController on essentially every navigation (7049 events), so
# treating it as a hub makes "return to root" fire constantly and shreds every
# session into 3-event fragments. Hubs must be screens the *user* recognises as
# home, not the container that hosts them.
ROOT_SCREENS: frozenset[str] = frozenset(
    {
        "HomeVC",
        "HomeGuestVC",
        "HOME",
        "android/Home",
        "guest/HOME",
    }
)

# Query-string keys that change *what the user sees* and must survive URL
# canonicalisation. Everything else (contractNo, orderId, timestamp, utm_*, ...)
# is an instance identifier and is dropped.
SEMANTIC_QUERY_KEYS: frozenset[str] = frozenset(
    {"tab", "cat_id", "categoryid", "ordertype", "type", "step", "mode", "view", "status"}
)

# Substrings that mark an action as a "go back" gesture — a core friction signal.
BACK_ACTION_MARKERS: tuple[str, ...] = (
    "btn_back",
    "backbutton",
    "handleback",
    "nav_back",
    "/back",
    "goback",
    "close",
    "dismiss",
    "cancel",
)


@dataclass
class CanonizeConfig:
    """Rules for turning raw rows into canonical (event_type, screen, target)."""

    # Namespace every screen by OS so that `HOME` (Android) can never collide
    # with `HOME` (iOS) even if both platforms ship the same string.
    namespace_by_os: bool = True
    # Strip non-semantic query params and mask id-like path segments in URLs.
    canonize_urls: bool = True
    # Mask digit-heavy / uuid-like segments inside slash-delimited names.
    mask_id_segments: bool = True
    # Retained for pickle compatibility only; production never uses duration.
    duration_clip_seconds: float = 1800.0
    # Depth at which action paths are truncated for the mid-resolution token.
    action_path_depth_mid: int = 3


@dataclass
class TokenConfig:
    """Vocabulary construction."""

    # Token resolution used to build feature vectors.
    #   L1 = screen only, L2 = screen + shallow action path, L3 = exact triple
    level: str = "L2"
    # Tokens observed in fewer than this many *journeys* fall back to their
    # coarser form, then to <RARE>. Guards against the 28% singleton tail.
    min_journey_df: int = 3


@dataclass
class PostProcessConfig:
    """Sequence cleanup applied after tokenisation."""

    collapse_consecutive: bool = True
    # Detect A-B-A-B... loops of period 2..max_cycle_period repeated at least
    # min_cycle_repeats times and collapse them (count kept as a feature).
    collapse_cycles: bool = True
    max_cycle_period: int = 4
    min_cycle_repeats: int = 2
    # Drop OS-chrome screens from the modelling stream (still counted).
    drop_chrome: bool = False
    # Drop boot/splash screens from the modelling stream (still counted).
    drop_boot: bool = False


@dataclass
class PrefixSpanConfig:
    """Frequent sequential pattern mining over journey sequences (route B)."""

    # Minimum share of journeys a pattern must appear in, with a floor so that
    # a small platform slice cannot admit patterns backed by 3 journeys.
    min_support: float = 0.02
    min_support_count: int = 20
    # Patterns longer than this explode combinatorially and stop generalising.
    max_pattern_length: int = 5
    # Length-1 patterns are token popularity, not behaviour.
    min_pattern_length: int = 2
    # Keep only closed patterns (no longer pattern with identical support).
    closed_only: bool = True
    # Cap on the emitted table / feature matrix width.
    max_patterns: int = 300


@dataclass
class SegmentConfig:
    """Session -> journey segmentation."""

    # Primary rule: idle gap. Data p97.5 = 73s, p99 = 306s.
    idle_gap_seconds: float = 90.0 # idle gap between events that triggers a new journey is 90 seconds.
    # Start a new journey when the user lands back on a root/hub screen.
    cut_on_root_return: bool = True
    # Require this many events since the last cut before a root return may cut.
    # Low values fragment sessions; 6 keeps hub-return meaningful.
    root_return_min_events: int = 6
    # Start a new journey on login/logout ACTIONS (not on merely viewing a
    # login screen — that fires on every guest browse).
    cut_on_auth_change: bool = True
    # Hard cap so pathological 1500-event sessions cannot form one journey.
    max_journey_length: int = 80
    # Journeys shorter than this are dropped from clustering (kept on disk).
    # Below 4 there is no order information left to cluster on.
    min_journey_length: int = 4
    # Optional statistical refinement layer (branching entropy). See segment.py.
    use_entropy_boundaries: bool = False
    entropy_percentile: float = 90.0
    entropy_ngram_order: int = 3


@dataclass
class FeatureConfig:
    """Journey representation."""

    ngram_range: tuple[int, int] = (1, 3)
    min_df: int = 3
    max_features: int = 20000
    sublinear_tf: bool = True
    svd_components: int = 64
    # Relative weight of the numeric/behavioural block vs the sequence block.
    numeric_block_weight: float = 0.35


@dataclass
class ClusterConfig:
    """Clustering + model selection."""

    method: str = "hdbscan"  # hdbscan | kmeans
    min_cluster_size: int = 100
    min_samples: int = 5
    cluster_selection_method: str = "eom"  # eom | leaf
    kmeans_k_grid: tuple[int, ...] = (6, 8, 10, 12, 15, 20, 25, 30)
    random_state: int = 42
    # Markov companion model used for likelihood-based anomaly scoring.
    fit_markov: bool = True
    markov_smoothing: float = 0.5


@dataclass
class PipelineConfig:
    input_csv: Path = Path("data/final_clean_events.csv")
    output_dir: Path = Path("outputs/journey")
    canonize: CanonizeConfig = field(default_factory=CanonizeConfig)
    tokens: TokenConfig = field(default_factory=TokenConfig)
    post: PostProcessConfig = field(default_factory=PostProcessConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    prefixspan: PrefixSpanConfig = field(default_factory=PrefixSpanConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)

    def to_dict(self) -> dict[str, Any]:
        def _norm(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _norm(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_norm(v) for v in value]
            return value

        return _norm(asdict(self))
