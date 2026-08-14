"""Stage 2 - Semantic enrichment (canonisation -> *here* -> tokenisation).

Canonisation makes an event *exact*; it does not make it *comparable*. After
stage 1 the two apps still describe one behaviour with two unrelated strings:

    iOS      view@FsListConnectedDeviceVC
    Android  view@internet_fprotect_screen/management_device/management_device_screen

Nothing downstream can see that these are the same screen, so an n-gram model
learns each platform's spelling separately and every long path becomes its own
singleton. This stage reads the *structure already present in the path* and
emits a coarse, controlled description of it:

    business_family      internet
    business_module      device
    business_object      device
    operation            view
    operation_stage      entry
    semantic_confidence  0.85
    semantic_evidence    family=internet<-screen:internet; ...

Both events above now agree on ``internet/device``, which is exactly the
generalisation the exact token cannot express.

How a label is derived
----------------------
* **View** events take their semantics from the screen path alone.
* **Action** events take *object* and *operation* from the action target, and
  *family*/*module* from the target when it names one, otherwise inherited from
  the screen the user was on. A target that is only a gesture (`btn_back`,
  `popup`, `confirm`) inherits everything but the operation.
* Nothing is forced. When neither the target nor the screen resolves a family,
  every slot stays `unknown` and `semantic_confidence` is 0.0 - a labelled
  `unknown` is information, a guessed label is damage.

Resolution rules, in full (they are the only two rules in this module):

* **section (family + module)** - the *deepest* match wins, because a path is
  written outside-in and the last section entered is the one the user is in.
  A match that pins a module outranks one that only pins the family, so
  ``.../internet_service_tab/modem_control`` resolves to ``internet/modem``
  rather than ``internet/general``.
* **object and operation** - the *deepest* match wins, ties broken by the
  longer phrase, because the verb and its noun live in the leaf segment:
  ``confirm_block_device`` is (block, device), not (confirm, device).

No rule mentions a URL, a screen name or a platform. Everything is expressed
over normalised components, aliases and the taxonomy in `taxonomy.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .taxonomy import (
    CONTAINER_SECTION,
    DYNAMIC_SEGMENT_PATTERNS,
    FALLBACK_FAMILY_MODULES,
    FAMILY_MODULES,
    GENERAL,
    GENERIC_ACTION_PHRASES,
    INFRASTRUCTURE_WORDS,
    OBJECT_PHRASES,
    OPERATION_PHRASES,
    OPERATION_STAGES,
    PHRASE_ALIASES,
    UNKNOWN,
    WEAK_FAMILY_MODULES,
    WORD_ALIASES,
)

SEMANTIC_COLUMNS: tuple[str, ...] = (
    "business_family",
    "business_module",
    "business_object",
    "operation",
    "operation_stage",
    "semantic_confidence",
    "semantic_evidence",
)

# Contribution of each resolved slot to `semantic_confidence`. Kept as a table
# so the score is auditable rather than an opinion buried in an expression.
CONFIDENCE_WEIGHTS: dict[str, float] = {
    "family_strong": 0.40,
    "family_weak": 0.20,
    "family_inherited": 0.15,
    "module_specific": 0.20,
    "module_general": 0.05,
    "module_inherited": 0.08,
    "object_matched": 0.15,
    "object_inherited": 0.07,
    "operation_matched": 0.25,
    "operation_default": 0.10,
}

# Split a path into segments, then a segment into parts, then a part into words.
_SEGMENT_SPLIT_RE = re.compile(r"[/\\]+")
_PART_SPLIT_RE = re.compile(r"[^0-9A-Za-z]+")
# `5GHz` and friends: a digit-led unit is one word, not "5" + "g" + "hz".
_UNIT_RE = re.compile(r"^\d+[A-Za-z]+$")
# camelCase / PascalCase / SCREAMING boundaries: FsListVC -> fs list vc
_WORD_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z][a-z0-9]*|[0-9]+")
# The iOS `VC` suffix fuses with a preceding acronym (`ManageAPVC` would split
# as `manage` + `apvc`), so it is removed before the word regex runs.
_VC_SUFFIX_RE = re.compile(r"(?<=[A-Za-z])VC$")


# --------------------------------------------------------------------------
# Phrase indexes, built and validated once at import
# --------------------------------------------------------------------------
def _phrase_words(phrase: str) -> tuple[str, ...]:
    return tuple(part for part in phrase.split("_") if part)


def _validate_phrase(phrase: str, words: tuple[str, ...], taxonomy: str) -> None:
    if not words:
        raise ValueError(f"{taxonomy}: empty phrase {phrase!r}")
    for word in words:
        if word in INFRASTRUCTURE_WORDS:
            raise ValueError(
                f"{taxonomy}: phrase {phrase!r} uses infrastructure word {word!r}, "
                "which is stripped before matching and can never match"
            )
        if word in WORD_ALIASES:
            raise ValueError(
                f"{taxonomy}: phrase {phrase!r} uses alias source {word!r}; "
                f"write it as {'_'.join(WORD_ALIASES[word])!r} instead"
            )
    if words in PHRASE_ALIASES:
        raise ValueError(
            f"{taxonomy}: phrase {phrase!r} is rewritten by PHRASE_ALIASES; "
            f"write it as {'_'.join(PHRASE_ALIASES[words])!r} instead"
        )


def _build_flat_index(
    taxonomy: dict[str, tuple[str, ...]], name: str
) -> dict[tuple[str, ...], str]:
    index: dict[tuple[str, ...], str] = {}
    for label, phrases in taxonomy.items():
        for phrase in phrases:
            words = _phrase_words(phrase)
            _validate_phrase(phrase, words, name)
            if words in index:
                raise ValueError(
                    f"{name}: phrase {phrase!r} maps to both {index[words]!r} and {label!r}"
                )
            index[words] = label
    return index


def _build_section_index(
    taxonomy: dict[str, dict[str, tuple[str, ...]]], name: str
) -> dict[tuple[str, ...], tuple[str, str]]:
    index: dict[tuple[str, ...], tuple[str, str]] = {}
    for family, modules in taxonomy.items():
        for module, phrases in modules.items():
            for phrase in phrases:
                words = _phrase_words(phrase)
                _validate_phrase(phrase, words, name)
                if words in index:
                    other = "/".join(index[words])
                    raise ValueError(
                        f"{name}: phrase {phrase!r} maps to both {other!r} and "
                        f"{family}/{module!r}"
                    )
                index[words] = (family, module)
    return index


_STRONG_SECTIONS = _build_section_index(FAMILY_MODULES, "FAMILY_MODULES")
_WEAK_SECTIONS = _build_section_index(WEAK_FAMILY_MODULES, "WEAK_FAMILY_MODULES")
_FALLBACK_SECTIONS = _build_section_index(FALLBACK_FAMILY_MODULES, "FALLBACK_FAMILY_MODULES")
_OBJECTS = _build_flat_index(OBJECT_PHRASES, "OBJECT_PHRASES")
_OPERATIONS = _build_flat_index(OPERATION_PHRASES, "OPERATION_PHRASES")

_SECTION_TIERS: tuple[tuple[str, dict[tuple[str, ...], tuple[str, str]]], ...] = (
    ("strong", _STRONG_SECTIONS),
    ("weak", _WEAK_SECTIONS),
    ("weak", _FALLBACK_SECTIONS),
)
if len({*_STRONG_SECTIONS} & {*_WEAK_SECTIONS} | {*_STRONG_SECTIONS} & {*_FALLBACK_SECTIONS}):
    raise ValueError("a phrase may appear in only one family tier")
if set(OPERATION_STAGES) - set(OPERATION_PHRASES):
    raise ValueError(
        "OPERATION_STAGES names operations absent from OPERATION_PHRASES: "
        f"{sorted(set(OPERATION_STAGES) - set(OPERATION_PHRASES))}"
    )

_MAX_SECTION_LEN = max(
    len(words) for words in (*_STRONG_SECTIONS, *_WEAK_SECTIONS, *_FALLBACK_SECTIONS)
)
_MAX_OBJECT_LEN = max(len(words) for words in _OBJECTS)
_MAX_OPERATION_LEN = max(len(words) for words in _OPERATIONS)
_MAX_ALIAS_LEN = max(len(words) for words in PHRASE_ALIASES)


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
_MISSING_TEXT = frozenset({"nan", "none", "null", "<na>", "nat"})


def _is_missing(raw: object) -> bool:
    """True for every way this pipeline spells "no value".

    `<missing>` / `<none>` are the canonisation sentinels; the bare words arrive
    when a frame round-trips through CSV or through `astype(str)`.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return True
    text = str(raw).strip()
    if not text or text.lower() in _MISSING_TEXT:
        return True
    return text.startswith("<") and text.endswith(">")


def _split_part(part: str) -> list[str]:
    if _UNIT_RE.match(part):
        return [part.lower()]
    return [word.lower() for word in _WORD_RE.findall(_VC_SUFFIX_RE.sub("", part))]


def _is_dynamic(text: str) -> bool:
    return any(pattern.match(text) for pattern in DYNAMIC_SEGMENT_PATTERNS)


def _apply_phrase_aliases(words: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    i = 0
    while i < len(words):
        for length in range(min(_MAX_ALIAS_LEN, len(words) - i), 0, -1):
            replacement = PHRASE_ALIASES.get(words[i : i + length])
            if replacement is not None:
                out.extend(replacement)
                i += length
                break
        else:
            out.append(words[i])
            i += 1
    return tuple(out)


@lru_cache(maxsize=8192)
def normalize_path(raw: str) -> tuple[str, ...]:
    """Turn any screen or action path into a canonical word stream.

    Handles, in one pass and without knowing a single concrete URL: slash
    paths, snake_case, camelCase/PascalCase/SCREAMING names, hosts, query
    strings, dynamic identifiers (`{id}`, digits, hex, `v920`, `null`) and
    infrastructure words. `hi.fpt.vn/web/shop/product-management-v920?cat_id=2`
    and `ProductManagementVC` both reduce to `('shop', 'product', 'management')`.
    """
    if _is_missing(raw):
        return ()
    text = str(raw).strip()
    # A query string only ever carried instance identifiers past canonisation.
    text = text.split("?", 1)[0].split("#", 1)[0]

    words: list[str] = []
    for segment in _SEGMENT_SPLIT_RE.split(text):
        if not segment or _is_dynamic(segment):
            continue
        for part in _PART_SPLIT_RE.split(segment):
            if not part or _is_dynamic(part):
                continue
            for word in _split_part(part):
                if _is_dynamic(word):
                    continue
                words.extend(WORD_ALIASES.get(word, (word,)))

    aliased = _apply_phrase_aliases(tuple(words))
    return tuple(word for word in aliased if word not in INFRASTRUCTURE_WORDS)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _Match:
    """One taxonomy hit: where it ended, how long it was, what it means."""

    end: int
    length: int
    phrase: tuple[str, ...]


def _find_matches(
    words: tuple[str, ...], index: dict[tuple[str, ...], object], max_length: int
) -> list[tuple[_Match, object]]:
    found: list[tuple[_Match, object]] = []
    for start in range(len(words)):
        for length in range(1, min(max_length, len(words) - start) + 1):
            phrase = words[start : start + length]
            value = index.get(phrase)
            if value is not None:
                found.append((_Match(start + length - 1, length, phrase), value))
    return found


def _deepest(candidates: list[tuple[_Match, object]]) -> tuple[_Match, object] | None:
    """Last match in the stream; ties broken by the longer phrase."""
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0].end, item[0].length))


def resolve_section(
    words: tuple[str, ...], *, allow_container: bool = False
) -> tuple[str, str, str, str] | None:
    """Return (family, module, evidence_phrase, strength) or None.

    Tiers are consulted in order - business taxonomy, then the `home` hub, then
    UI chrome - so a popup inside the payment flow stays payment and only a
    path with no business meaning at all falls through to `chrome`. Within a
    tier the deepest match wins, and a match that pins a module outranks one
    that only pins the family.

    `allow_container` enables the last resort, and only screens get it: a
    *screen* whose path is entirely infrastructure is a navigation container,
    but an action *target* like `btn_back` is a gesture that must inherit the
    screen it fired on rather than declare itself chrome.
    """
    for strength, index in _SECTION_TIERS:
        matches = _find_matches(words, index, _MAX_SECTION_LEN)
        if not matches:
            continue
        specific = [item for item in matches if item[1][1] != GENERAL]
        best = _deepest(specific) or _deepest(matches)
        assert best is not None  # matches is non-empty here
        family, module = best[1]
        return family, module, "_".join(best[0].phrase), strength

    # Everything was stripped as infrastructure, or nothing survived but bare
    # verbs (`HiWebViewActivity` -> ("view",)). That is a container, not a
    # mystery: there is no business noun left to be wrong about.
    if allow_container and all((word,) in _OPERATIONS for word in words):
        family, module = CONTAINER_SECTION
        return family, module, "container_only", "weak"
    return None


def resolve_object(words: tuple[str, ...]) -> tuple[str, str] | None:
    """Return (business_object, evidence_phrase) or None."""
    best = _deepest(_find_matches(words, _OBJECTS, _MAX_OBJECT_LEN))
    return (str(best[1]), "_".join(best[0].phrase)) if best else None


def resolve_operation(words: tuple[str, ...]) -> tuple[str, str] | None:
    """Return (operation, evidence_phrase) or None."""
    best = _deepest(_find_matches(words, _OPERATIONS, _MAX_OPERATION_LEN))
    return (str(best[1]), "_".join(best[0].phrase)) if best else None


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SemanticLabel:
    """The coarse meaning of one event. Every field is a controlled value."""

    business_family: str = UNKNOWN
    business_module: str = UNKNOWN
    business_object: str = UNKNOWN
    operation: str = UNKNOWN
    operation_stage: str = UNKNOWN
    semantic_confidence: float = 0.0
    semantic_evidence: str = ""

    @property
    def is_known(self) -> bool:
        return self.business_family != UNKNOWN


UNKNOWN_LABEL = SemanticLabel()


class _LabelBuilder:
    """Accumulates slots, evidence and confidence for one event."""

    def __init__(self) -> None:
        self.family = UNKNOWN
        self.module = UNKNOWN
        self.object = UNKNOWN
        self.operation = UNKNOWN
        self.confidence = 0.0
        self.evidence: list[str] = []

    def credit(self, weight_key: str, slot: str, value: str, source: str, phrase: str) -> None:
        self.confidence += CONFIDENCE_WEIGHTS[weight_key]
        self.evidence.append(f"{slot}={value}<-{source}:{phrase}")

    def build(self) -> SemanticLabel:
        if self.family == UNKNOWN:
            return UNKNOWN_LABEL
        return SemanticLabel(
            business_family=self.family,
            business_module=self.module,
            business_object=self.object,
            operation=self.operation,
            operation_stage=OPERATION_STAGES.get(self.operation, UNKNOWN),
            semantic_confidence=round(min(self.confidence, 1.0), 3),
            semantic_evidence=";".join(self.evidence),
        )


def _apply_section(
    builder: _LabelBuilder, words: tuple[str, ...], source: str, *, allow_container: bool = False
) -> bool:
    section = resolve_section(words, allow_container=allow_container)
    if section is None:
        return False
    family, module, phrase, strength = section
    builder.family = family
    builder.module = module
    builder.credit(f"family_{strength}", "family", family, source, phrase)
    builder.credit(
        "module_general" if module == GENERAL else "module_specific",
        "module",
        module,
        source,
        phrase,
    )
    return True


@lru_cache(maxsize=8192)
def classify_screen(screen: str) -> SemanticLabel:
    """Semantics of a View event, taken from the screen path alone."""
    if _is_missing(screen):
        # No screen was recorded. An absent path is absent evidence, and the
        # container fallback below must not turn it into `chrome`.
        return UNKNOWN_LABEL
    words = normalize_path(screen)
    builder = _LabelBuilder()
    if not _apply_section(builder, words, "screen", allow_container=True):
        return UNKNOWN_LABEL

    matched_object = resolve_object(words)
    if matched_object is not None:
        builder.object = matched_object[0]
        builder.credit("object_matched", "object", matched_object[0], "screen", matched_object[1])

    matched_operation = resolve_operation(words)
    if matched_operation is not None:
        builder.operation = matched_operation[0]
        builder.credit("operation_matched", "operation", matched_operation[0], "screen", matched_operation[1])
    else:
        # A screen the user is looking at *is* a view; that is observation, not
        # a guess, so it is credited less than a verb read off the path.
        builder.operation = "view"
        builder.credit("operation_default", "operation", "view", "screen", "event_type")
    return builder.build()


@lru_cache(maxsize=8192)
def classify_action(target: str, context: SemanticLabel) -> SemanticLabel:
    """Semantics of an Action event.

    Object and operation come from the action target. Family and module come
    from the target when it names a section, otherwise from `context` - the
    label of the screen the action happened on. A purely gestural target
    (`btn_back`, `popup`, `confirm`) inherits family, module *and* object.
    """
    words = normalize_path(target)
    builder = _LabelBuilder()
    generic = words in GENERIC_ACTION_PHRASES

    if generic or not _apply_section(builder, words, "target"):
        if context.is_known:
            builder.family = context.business_family
            builder.module = context.business_module
            builder.credit(
                "family_inherited", "family", context.business_family, "context", "screen_context"
            )
            builder.credit(
                "module_inherited", "module", context.business_module, "context", "screen_context"
            )

    matched_object = None if generic else resolve_object(words)
    if matched_object is not None:
        builder.object = matched_object[0]
        builder.credit("object_matched", "object", matched_object[0], "target", matched_object[1])
    elif context.business_object != UNKNOWN:
        builder.object = context.business_object
        builder.credit(
            "object_inherited", "object", context.business_object, "context", "screen_context"
        )

    matched_operation = resolve_operation(words)
    if matched_operation is not None:
        builder.operation = matched_operation[0]
        builder.credit("operation_matched", "operation", matched_operation[0], "target", matched_operation[1])
    return builder.build()


def classify_event(event_type: str, screen_context: str, segment_name: str) -> SemanticLabel:
    """Label one canonical event using the role its columns play.

    `event_type` decides which column is the screen: for a View the screen is
    `screen_context`, for an Action `screen_context` is the screen the user was
    already on and `segment_name` is what they touched.
    """
    context = classify_screen(screen_context)
    if str(event_type).lower() == "view":
        return context
    return classify_action(str(segment_name), context)


# --------------------------------------------------------------------------
# Frame-level stage
# --------------------------------------------------------------------------
def annotate_semantics(canon: pd.DataFrame) -> pd.DataFrame:
    """Attach `SEMANTIC_COLUMNS` to a canonical event frame.

    Source columns are never modified; the stage only adds. Labels are resolved
    once per distinct (event_type, screen_context, segment_name) triple, which
    is a few thousand rows even on a multi-million-event corpus.
    """
    required = {"event_type", "screen_context", "segment_name"}
    missing = required - set(canon.columns)
    if missing:
        raise ValueError(f"cannot enrich; canonical frame is missing {sorted(missing)}")

    keys = ["event_type", "screen_context", "segment_name"]
    distinct = canon[keys].astype(str).drop_duplicates().reset_index(drop=True)
    labels = [
        classify_event(row.event_type, row.screen_context, row.segment_name)
        for row in distinct.itertuples(index=False)
    ]
    for column in SEMANTIC_COLUMNS:
        distinct[column] = [getattr(label, column) for label in labels]

    out = canon.copy()
    merged = out[keys].astype(str).merge(distinct, on=keys, how="left", sort=False)
    for column in SEMANTIC_COLUMNS:
        out[column] = merged[column].to_numpy()
    return out


def semantic_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Coverage of the enrichment stage: what got labelled, and how well."""
    total = len(frame)
    if total == 0:
        return pd.DataFrame(columns=["metric", "value"])

    known = frame["business_family"].ne(UNKNOWN)
    specific_module = known & frame["business_module"].ne(GENERAL)
    rows: list[dict[str, object]] = [
        {"metric": "events", "value": int(total)},
        {"metric": "family_coverage", "value": round(float(known.mean()), 4)},
        {"metric": "family_unknown_rate", "value": round(float((~known).mean()), 4)},
        {"metric": "module_specific_rate", "value": round(float(specific_module.mean()), 4)},
        {
            "metric": "object_unknown_rate",
            "value": round(float(frame["business_object"].eq(UNKNOWN).mean()), 4),
        },
        {
            "metric": "operation_unknown_rate",
            "value": round(float(frame["operation"].eq(UNKNOWN).mean()), 4),
        },
        {
            "metric": "mean_confidence",
            "value": round(float(frame["semantic_confidence"].mean()), 4),
        },
        {
            "metric": "low_confidence_rate",
            "value": round(float(frame["semantic_confidence"].lt(0.5).mean()), 4),
        },
        {"metric": "distinct_families", "value": int(frame.loc[known, "business_family"].nunique())},
    ]
    return pd.DataFrame(rows)


def unknown_examples(frame: pd.DataFrame, top_k: int = 25) -> pd.DataFrame:
    """The highest-traffic events the taxonomy could not read.

    This is the manual-review queue: every row here is either a genuinely
    meaningless event or one missing phrase in `taxonomy.py`.
    """
    unknown = frame.loc[frame["business_family"].eq(UNKNOWN)]
    if unknown.empty:
        return pd.DataFrame(columns=["event_type", "screen_context", "segment_name", "events"])
    return (
        unknown.groupby(["event_type", "screen_context", "segment_name"], sort=False)
        .size()
        .reset_index(name="events")
        .sort_values("events", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    canonical_df_path = "output/journey_runs/_prepared/data_raw_sample-0fc0a9ec_time-session-split_test0p2_gap90_max80_root6_auth1/canonical_events_android.csv"
    canon = pd.read_csv(canonical_df_path)
    enriched = annotate_semantics(canon)
    enriched.to_csv("output/journey_runs/_prepared/data_raw_sample-0fc0a9ec_time-session-split_test0p2_gap90_max80_root6_auth1/canonical_events_android_semantics.csv", index=False)