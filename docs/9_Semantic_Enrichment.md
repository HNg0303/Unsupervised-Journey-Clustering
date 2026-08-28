# Semantic Enrichment — from exact tokens to comparable ones

## 0. The problem this stage exists to solve

After the first complete run, the modelling token was the exact canonical event:

```
view@internet_fprotect_screen/management_device/management_device_screen
view@FsListConnectedDeviceVC
```

Those two rows are the **same screen** — Android and iOS device management. The
model cannot know that. Two consequences, both visible in the July run:

* **The vocabulary is a long tail.** ~9 000 distinct exact tokens across the two
  platforms, of which two thirds occur in fewer than five journeys. An n-gram
  over that is mostly singletons.
* **Near-synonyms never meet.** `L1`, `L2` and `L3` were three *aliases* of the
  exact token, so `--level L1` changed nothing and the rare-token backoff had
  nowhere to back off to.

The fix is a stage between canonisation and tokenisation that reads the
structure already present in the path and emits a coarse, controlled
description of it.

## 1. Where it sits

```
raw events
  → 1 canonize        role-correct (event_type, screen, target); URL/id masking
  → 2 ENRICH          business_family / module / object / operation / stage
  → 3 tokenize        exact + L1 + L2 + L3 + operation channels
  → 4 segment         sessions → journeys
  → 5 postprocess     collapse runs and cycles; numeric block
  → 6 features        one TF-IDF+SVD block per channel, weighted concat
  → 7 cluster         HDBSCAN + KMeans baseline
```

Stage 2 only **adds** columns. No source event column is modified, and the exact
token is unchanged, so every existing artefact (`sequence`, the shareholder
catalogs, the mobile bundle) still means what it meant.

## 2. What it emits

`src/Rule_based/semantics.py` adds seven columns per event:

| column | example | meaning |
|---|---|---|
| `business_family` | `internet` | top-level product area |
| `business_module` | `modem` | sub-area within the family |
| `business_object` | `modem` | the noun the interaction is about |
| `operation` | `restart` | the verb |
| `operation_stage` | `commit` | `entry`/`browse`/`configure`/`commit`/`abort` |
| `semantic_confidence` | `1.0` | 0–1, from an auditable weight table |
| `semantic_evidence` | `family=internet<-screen:modem;…` | which phrase decided each slot |

## 3. How a label is derived

**Normalisation** (`normalize_path`) turns any path into a word stream, without
knowing a single concrete URL: it splits on slashes, then on `_`/`-`/`.`, then on
camelCase/PascalCase/SCREAMING boundaries; drops query strings, hosts, dynamic
identifiers (`{id}`, digits, hex, `v920`, `null`) and infrastructure words;
applies word aliases (`fprotect`→`fsafe`, `payement`→`payment`, `ap`→`access
point`) and phrase aliases (`wi fi`→`wifi`, `turn on off`→`toggle`).

```
hi.fpt.vn/web/shop/product-management-v920?cat_id=2  →  ('shop','product','management')
ProductManagementVC                                   →  ('shop','product','management')
```

**Matching** against `src/Rule_based/taxonomy.py`, by two rules only:

* **section (family + module)** — tiers are consulted in order: business
  taxonomy, then the `home` hub, then UI `chrome`. Within a tier the *deepest*
  match wins and a match that pins a module outranks one that only pins the
  family. A path is written outside-in, so the last section entered is the one
  the user is in: `.../internet_service_tab/modem_control` → `internet/modem`,
  not `internet/general`.
* **object and operation** — deepest match wins, ties broken by the longer
  phrase, because the verb and its noun live in the leaf:
  `confirm_block_device` is (block, device), not (confirm, device).

**View vs Action**

* A **View** takes its semantics from the screen path alone.
* An **Action** takes object and operation from the target, and family/module
  from the target when it names a section — otherwise inherited from the screen
  the user was on.
* A purely gestural target (`btn_back`, `popup`, `confirm`, `/`) inherits family,
  module *and* object. `btn_back` on the modem screen is "left the modem
  screen", not a fifth kind of event.
* When neither the target nor the screen resolves a family, every slot stays
  `unknown` and confidence is 0.0. **A labelled `unknown` is information; a
  guessed label is damage.**

`chrome` is a real answer, not a fallback guess: `MainTabBarController`,
`SplashActivity` and `PopupVC` are navigation containers the user never chose.
It is consulted last, so a popup *inside* the payment flow stays `payment`.

## 4. The taxonomy is validated at import

`taxonomy.py` is data. `semantics.py` rejects it at import time if a phrase
collides with another entry, contains an infrastructure word (it could never
match), or is not in alias-canonical form (it would be rewritten before
matching). A broken taxonomy fails loudly instead of silently mislabelling half
the corpus.

## 5. The token ladder

One event, five tokens:

| channel | column | example | vocabulary (July, both platforms) |
|---|---|---|---|
| exact | `exact_token` | `action@ManageModemVC#do_action/MODEM_RESET` | ~9 000 | 
| intent | `token_l3` | `internet/modem/modem/restart` | 985 |
| coarse | `token_l2` | `internet/modem` | 50 |
| family | `token_l1` | `internet` | 14 |
| operation | `operation_token` | `commit:restart` | ~35 |

`--level` selects the primary modelling token (`EXACT` by default);
`--backoff-level` selects what the rare-token backoff degrades to. The backoff
now *means* something: a one-off deep path degrades to `internet/modem` instead
of straight to `<rare>`.

## 6. Multi-channel features

`JourneyVectorizer` gives each channel its own TF-IDF + SVD block, L2-normalises
it, scales it by its weight and concatenates. Normalising *before* weighting is
what makes a channel's influence its weight and nothing else — not its
vocabulary size, not its SVD rank.

```bash
python scripts/run_partitioned_journey_pipeline.py --channel-weights "coarse=0.45,intent=0.30,operation=0.20" 0.5, 0.5
```

`--channel-weights ""` reproduces the single-channel representation exactly.
That is required for the ONNX/mobile bundle: on-device preprocessing rebuilds
one TF-IDF block plus the numeric block, so the current full-data scorer validates
the representation before shipping features that silently disagree with
the centroids.

## 7. Coverage on the July corpus

2 027 469 events, both platforms:

| metric | value |
|---|---|
| family coverage | 99.9 % |
| family unknown rate | 0.1 % |
| specific (non-`general`) module | 65 % |
| operation unknown rate | 4 % |
| object unknown rate | 44 % |
| mean confidence | 0.70 |

`semantics.unknown_examples()` is the manual-review queue: the highest-traffic
events the taxonomy could not read. Each row is either a genuinely meaningless
event or one missing phrase in `taxonomy.py`.

## 8. Prior art

This is event/activity abstraction as practised in process mining — mapping
low-level log events onto a controlled activity ontology before discovery, so
that models generalise over instrumentation rather than memorising it. The
analytics equivalent is *content grouping* / screen taxonomies in product
analytics tools. The multi-resolution part is standard back-off n-gram
modelling; the multi-channel part is ordinary multi-view concatenation. Nothing
here is novel — the value is that the taxonomy is small, validated and specific
to this app.
