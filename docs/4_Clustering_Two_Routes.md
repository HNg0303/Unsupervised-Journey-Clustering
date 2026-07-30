# Journey Clustering — Two Representation Routes

How a cleaned clickstream sequence becomes a cluster label, by two independent
paths; what those clusters actually look like on HiFPT Android and iOS data; how
to score new events; and how to look at the result.

All numbers below are from the run committed alongside this document
(`outputs/clusters/`), token level **L2**, `min_cluster_size = 15`.

---

## 0. Where this sits

```
raw events → canonize → tokenize (L1/L2/L3) → segment into journeys → postprocess
                                                                          │
                                              ┌───────────────────────────┴──────┐
                                     ROUTE A  │                                  │  ROUTE B
                             TF-IDF n-grams   ▼                                  ▼   PrefixSpan
                                        SVD(64)                             SVD(64)
                                              └──────────┬───────────────────────┘
                                                    + numeric block (10 features, weight 0.35)
                                                         │
                                                     HDBSCAN
                                                         │
                                              per-cluster Markov chain
                                                         │
                                          cluster · anomaly score · next action
```

Both routes consume the **same** postprocessed input and append the **same**
numeric block, so any difference in the clusters is caused by the sequence
encoding alone. That is the point of running both.

Input per platform after Stage 4:

| | journeys | events (cleaned) | median length |
|---|---|---|---|
| Android | 1,795 | 18,804 | 7 |
| iOS | 4,063 | 42,639 | 6 |

---

## 1. Route A — TF-IDF over n-grams

### Algorithm

Treat each journey as a short document whose "words" are tokens, then count
**contiguous** runs of 1 to N tokens.

1. Join the sequence with sentinels: `<bos> A B C <eos>`. The sentinels matter —
   they make "started at Home" and "ended at Payment" first-class features rather
   than accidents of position.
2. Enumerate every contiguous n-gram for n = 1…4. `A B C` yields
   `A`, `B`, `C`, `A B`, `B C`, `A B C`.
3. TF-IDF weight them. Sub-linear TF (`1 + log tf`) stops a token repeated 20
   times inside one journey from dominating; IDF down-weights the tokens every
   journey contains (`View@iOS::HomeVC` is in a quarter of them and says nothing).
4. `min_df = 3` drops n-grams seen in fewer than 3 journeys — mostly the
   singleton tail.
5. Truncated SVD to 64 dimensions, then L2-normalise. SVD is used rather than PCA
   because the matrix is sparse and must not be densified.

### Implementation, plainly

[features.py](../src/Rule_based/features.py) — `JourneyVectorizer`.

Tokens are opaque strings containing `/`, `#`, `:` and `?`, so the usual word
regex would shred them. The analyzer is therefore `\S+`: join tokens with spaces,
split on whitespace, nothing else. `lowercase=False`, because
`Action@iOS::LoginVC` and `action@ios::loginvc` should never merge.

```python
vec = JourneyVectorizer(cfg.features)      # cfg.features.ngram_range = (1, 4)
matrix, info = vec.fit_transform(journeys, sequences)
```

`fit_transform` learns the vocabulary, SVD basis and numeric scaler; `transform`
reuses them. That split is the whole reason new data can be scored later.

### Does N=4 help?

Measured, not assumed (`--ngram-sweep`):

| platform | n-grams | vocabulary | SVD var. | clusters | noise | silhouette | DB |
|---|---|---|---|---|---|---|---|
| Android | 1–2 | 1,540 | 0.546 | 23 | 34.0% | 0.336 | 1.376 |
| Android | 1–3 | 2,849 | 0.457 | 26 | 38.3% | 0.354 | 1.344 |
| **Android** | **1–4** | **3,968** | 0.417 | **29** | 38.7% | **0.357** | **1.227** |
| iOS | 1–2 | 2,847 | 0.471 | 50 | 41.3% | **0.389** | 1.089 |
| iOS | 1–3 | 5,665 | 0.375 | 53 | 41.1% | 0.379 | 1.099 |
| **iOS** | **1–4** | **8,005** | 0.334 | 50 | **39.1%** | 0.377 | 1.098 |

Reading it honestly: **4-grams help Android slightly and do nothing for iOS.**
Android gains 3 clusters and the best Davies-Bouldin; iOS pays 2.8× the
vocabulary for a silhouette that moves by 0.012 in the wrong direction. The
default is `(1, 4)` because the cost is only memory, but `(1, 3)` is defensible
and `(1, 2)` is what to use if the vocabulary ever becomes a problem.

Note the SVD explained variance *falls* as N rises (0.55 → 0.42 Android). That is
expected, not a regression: longer n-grams add rare, near-orthogonal columns that
64 components cannot absorb. Judge by cluster quality, not by that number.

---

## 2. Route B — PrefixSpan sequential patterns

### Algorithm

An n-gram must be **contiguous**. A PrefixSpan pattern need not be. That single
difference is the entire motivation:

```
journey  : HOME → SUPPORT → BACK → PAY → CONFIRM
n-gram   : "HOME PAY CONFIRM" does NOT match — the detour broke it
pattern  : HOME → PAY → CONFIRM DOES match — it is a subsequence
```

Real users detour constantly, so route A sees the *exact path* while route B sees
the *goal*.

PrefixSpan (Pei et al., 2001) grows patterns depth-first with pseudo-projection:

1. Start with the empty pattern; its projection is every journey at position 0.
2. For each candidate next token, find its **first** occurrence at or after the
   current position in each projected journey. First occurrence only — that makes
   support count *journeys*, not occurrences, so one user looping 200 times
   cannot manufacture a pattern.
3. Keep the token if it appears in ≥ `min_support` journeys. Emit the extended
   pattern; recurse on the projection it induces.
4. Stop at `max_pattern_length` (5).

The projection is a list of `(journey_id, position)` pairs — no database copies,
which is what "pseudo" means and why this is fast enough to run inline.

**Closed-pattern filtering.** A pattern is *closed* when no longer pattern has
identical support. Without this the output is drowned in every prefix of the same
shape (`A`, `A→B`, `A→B→C` all with support 79). Only the maximal one carries
information.

### Implementation, plainly

[prefixspan.py](../src/Rule_based/prefixspan.py) — about 90 lines of actual
algorithm.

```python
patterns = mine_patterns(sequences, cfg.prefixspan)   # DataFrame of patterns
features = pattern_features(sequences, patterns)      # journey × pattern binary
candidates = journey_candidates(sequences, patterns)  # rule-based archetype guess
```

Because each event is a single item (not an itemset), there is no
itemset-extension step — only sequence extension. That is why the core is short.

One tuning detail worth knowing: **the `max_patterns` cap is spent per length,
not globally.** Ranking purely by length keeps only length-5 patterns, and
coverage collapsed to 30% / 9%. Per-length quotas put it at **90.8% Android /
78.6% iOS** of journeys matching at least one pattern.

Mined at `min_support = 2%` (floor 20 journeys), capped at 300 patterns:

| platform | patterns | mean patterns/journey | journeys matching none |
|---|---|---|---|
| Android | 300 | 16.7 | 165 (9.2%) |
| iOS | 300 | 16.0 | 870 (21.4%) |

### Two exits from route B

**B-1, features.** `PatternVectorizer` exposes the *identical* API as
`JourneyVectorizer`, so everything downstream — clustering, scorer, inference —
accepts either without knowing which it got.

**B-2, journey candidates.** Label each journey with the most specific pattern it
contains (longest, ties broken by rarest). This is an archetype guess with **no
clustering at all** — a cheap, fully interpretable baseline and a sanity check on
the clusters you do fit. Written to `{platform}_journey_candidates.csv`.

### The trap, and how it is handled

A journey matching **zero** patterns has an all-zero sequence block. Every such
journey sits on the same point, and HDBSCAN dutifully reports them as one huge
cluster — 165 on Android, 870 on iOS, exactly the unmatched counts. That is not
an archetype, it is "no evidence". `run_clustering.py` relabels them as noise and
re-scores; the numbers in §3 are post-fix. **Route B's noise share is therefore
the honest one, and it is higher than route A's on iOS.**

---

## 3. Results

### Head to head

| platform | route | clusters | noise | silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|---|---|
| Android | A tf-idf | 29 | 38.7% | **0.357** | 1.227 | 61.9 |
| Android | B prefixspan | 34 | **35.5%** | 0.337 | **1.158** | **99.8** |
| iOS | A tf-idf | 50 | **39.1%** | 0.377 | **1.098** | 122.3 |
| iOS | B prefixspan | 56 | 43.6% | **0.402** | 1.109 | **217.6** |

Both routes find **many small archetypes, not a handful of big ones**, and both
leave 35–44% of journeys unassigned. For behavioural clickstream data that is a
reasonable outcome, not a failure: HDBSCAN refusing to force a label is exactly
why it was chosen, and that noise set is the first anomaly signal.

Route B scores better on Calinski-Harabasz by a wide margin (1.6–1.8×) and its
SVD retains 97% of variance versus 33–42% for route A — 300 binary pattern
columns are simply far more compressible than 8,000 TF-IDF columns.

### Do the routes agree?

| platform | ARI (all) | ARI (non-noise only) |
|---|---|---|
| Android | 0.131 | **0.577** |
| iOS | 0.072 | **0.513** |

This is the most informative number in the report. Overall agreement looks poor,
but that is dominated by disagreement about *what counts as noise*. Restricted to
journeys both routes clustered, agreement is **0.51–0.58** — substantial.

So: **the two routes broadly agree on the core structure and disagree on the
fringe.** Journeys clustered by both, into corresponding clusters, are solid.
Journeys the routes disagree about are exactly the ones worth reviewing by hand.

### What the archetypes actually are — Android (route A)

| # | n | med. len | med. span | back | loops | what it is |
|---|---|---|---|---|---|---|
| 25 | 129 | 11 | 42 s | 0.05 | 16% | **Guest bill payment** — Home → guest home → dismiss invite popup → billing tab → payment |
| 8 | 121 | 7 | 8 s | 0.12 | 2% | **Contract switching** — contract list → change contract → choose contract. Fast, mechanical |
| 19 | 90 | 21 | 88 s | 0.16 | 26% | **Internet service management** — long, loopy, high revisit (0.41). Friction candidate |
| 26 | 75 | 8 | 35 s | 0.05 | 11% | **Postpaid bill payment** — payment nav → select bill → pay |
| 18 | 60 | 7 | 7 s | 0.09 | 8% | **Prepaid payment bounce** — enters prepaid, hits back, returns. Short and abandoned |
| 20 | 47 | 24 | 126 s | 0.10 | 19% | **E-contract review** — longest common flow, contract list → unconfirmed e-contract |
| 1 | 44 | 4 | 13 s | 0.00 | 0% | **OAuth login** — login → AuthorizationManagement → RedirectUriReceiver. Zero back, zero revisit: clean |
| 24 | 44 | 6 | 7 s | 0.18 | 2% | **Support request open/close** — highest back rate of the top group |

### What the archetypes actually are — iOS (route A)

| # | n | med. len | med. span | back | loops | what it is |
|---|---|---|---|---|---|---|
| 26 | 195 | 5 | 7 s | 0.03 | 4% | **Guest banner → login wall** — taps banner, gets `PopupInviteLoginVC`, returns to guest home |
| 47 | 188 | 21 | 78 s | 0.04 | **20%** | **Internet package update** — long webview flow, revisit 0.44. The iOS twin of Android #19 |
| 8 | 167 | 7 | 5 s | 0.03 | 12% | **Support browsing** — cycles support description ↔ request list |
| 10 | 159 | 5 | 6 s | 0.01 | 1% | **OTP login** — continue_login → login_with_otp → OTPCreatePin. Cleanest cluster in the set |
| 32 | 138 | 5 | 3 s | **0.20** | 0% | **Immediate back-out** — Home → back → service manage → back. Highest back rate |
| 28 | 109 | 5 | 4 s | 0.02 | 9% | **Reminder popup skip** — popup shown, `skip_remind` tapped |
| 31 | 82 | 6 | 7 s | 0.01 | 2% | **Payment → promotion browse** |
| 39 | 81 | 17 | 74 s | 0.01 | **52%** | **E-contract tab thrash** — over half these journeys contain a detected loop |

**Cross-platform reading.** The same behaviours appear on both platforms
(payment, login, support, service management), but iOS fragments into ~1.7× more
clusters for 2.3× the journeys — its webview-heavy flows (`staging-hi.fpt.vn/...`)
generate more distinct token vocabulary. Modelling the platforms separately was
the right call: `HomeVC` and `android/Home` are the same user intent with zero
string overlap, and a joint model would have spent its capacity re-learning the
platform split.

**Flags worth acting on.** Android #19 and iOS #47/#39 are the same story:
long, high-revisit, loop-heavy service-management flows. iOS #39 has loops in
52% of its journeys. These are where to look first for a real UX problem.

---

## 4. Inference on new clickstream data

### What is fitted vs. what is re-run

`JourneyScorer` ([score.py](../src/Rule_based/score.py)) pickles the *entire*
fitted state: TF-IDF vocabulary, SVD basis, numeric scaler, cluster centroids,
Markov chains and calibrated thresholds. On new data it **re-runs the identical
pipeline itself** — canonize → tokenize → segment → cleanup → vectorize — so
training and inference cannot drift apart. Nothing is re-fitted.

The input must be **raw events** in the training schema:

```
record_id, device_id, customer_id, session_id, created_at, timestamp,
key, segmentation.name, segmentation.segment, segmentation.screen_id, duration
```

### Running it

```bash
.venv/Scripts/python scripts/score_new_events.py --platform android --input data/new_events.csv
```

Smoke test against the last day of the training file:

```bash
.venv/Scripts/python scripts/score_new_events.py --platform ios --holdout-days 1 --predict-next
```

From Python:

```python
from Rule_based.score import JourneyScorer
scorer = JourneyScorer.load("outputs/clusters/android_scorer.pkl")
scored = scorer.score(pd.read_csv("data/new_events.csv"))
```

### What comes back

One row per journey: `cluster`, `distance_to_centroid`, `markov_logprob`,
`geometric_anomaly`, `generative_anomaly`, `severe_anomaly`, `friction_flags`,
`sequence`, `next_action`, `next_action_share`.

**Two independent anomaly channels, kept separate on purpose:**

- **Geometric** — far from every centroid (beyond the training p95 distance).
  "This journey has an unusual *shape*."
- **Generative** — low Markov log-probability under its assigned archetype
  (below training p05). "The *transitions inside it* are improbable."

They disagree often enough that blending them into one number would destroy
information. `severe_anomaly` = both.

Measured on the last calendar day held out from training:

| | Android | iOS |
|---|---|---|
| journeys scored | 219 | 642 |
| matched a known archetype | **95.9%** | **93.3%** |
| geometric anomalies | 4.1% | 6.7% |
| generative anomalies | 20.5% | 18.4% |
| severe (both) | 0 | 1 |

93–96% of unseen journeys land on a known archetype, which is the number that
says the clusters generalise rather than memorise.

`friction_flags` explains *why* a journey looks bad, in words rather than a
score — `excessive_back`, `navigation_loop`, `screen_thrash`, `slow_journey`,
`unknown_archetype`, `improbable_transitions`. Android holdout, most common
first: `improbable_transitions` (45), `screen_thrash` (27), `navigation_loop`
(21), `excessive_back` (12), `slow_journey` (10).

### Predicting the next action

The same per-cluster Markov chain that produces the anomaly score, read forwards:

```python
scorer.predict_next(sequence_so_far, cluster=8, top_k=3)
```

```
  J000019  cluster=8
    at        : Action@iOS::/dkol/update-package/home#webHeader/BackButton
    ->  40.0%  View@iOS::staging-hi.fpt.vn/dkol/update-package/home
    ->  20.0%  Action@iOS::.../product-management-v920?cat_id=3#web/LocationButton
    ->  20.0%  View@iOS::nointent_pop-up
```

Conditioning on the cluster is what makes this useful — `P(next | screen)` is
very different for "paying a bill" than for "hunting for support", and the global
chain would average them into mush.

Two caveats, stated plainly:

- It is **first order**: only the last token conditions the prediction. Enough
  for next-screen guessing, not for multi-step intent.
- Two probabilities are reported. `smoothed_probability` is what the anomaly
  score uses, but with V ≈ 700 tokens the smoothing mass (`s·V` ≈ 352) dominates
  the denominator and squashes every value toward zero. **Quote
  `observed_share`** to humans — it is the plain "of users who reached this
  screen, 40% went to X next".

### Retraining

Nothing here adapts online. Re-run `run_clustering.py` when the app ships
navigation changes or when `matched a known archetype` on fresh data drops below
~85% — that ratio is the drift monitor.

---

## 5. Visualising the clusters

### What ships now — zero dependencies

```bash
.venv/Scripts/python scripts/visualize_clusters.py --platform android --route a
.venv/Scripts/python scripts/visualize_clusters.py --platform ios --route b
```

Writes a **self-contained HTML** file — no CDN, no plotting library, no server.
Canvas scatter coloured by cluster; hover any point to read its full token
sequence; click a legend entry to isolate a cluster. Output:
`outputs/clusters/{platform}_{route}_map.html`.

The hover matters more than the plot. The useful question is never "where are the
clusters" but "what is *in* that blob", and that needs the sequence attached to
every point.

### Recommended packages

None of these are installed in `.venv` yet; all are commented into
`requirements.txt`.

| package | use | why |
|---|---|---|
| **umap-learn** | 2D projection | The one that matters. Preserves global structure far better than t-SNE on sparse clustered embeddings. `visualize_clusters.py` picks it up automatically if installed — no code change |
| **plotly** | interactive scatter, Sankey | Sankey is the right chart for clickstream: screen-to-screen flow volumes per cluster. Nothing else shows a journey *as* a journey |
| **matplotlib + seaborn** | static figures | For the report/deck. Heatmap of pattern × cluster membership is the single most informative static view of route B |
| **datamapplot** | labelled cluster maps | Publication-quality maps with cluster labels placed automatically. Best-looking single artefact if you need one slide |
| **hdbscan** (standalone) | condensed tree plot | `plot.condensed_tree()` shows *why* HDBSCAN chose the clusters it chose — worth it once, while tuning `min_cluster_size` |

```bash
pip install umap-learn plotly
```

Once UMAP is installed, re-running `visualize_clusters.py` uses it automatically
and the projection improves with no other change.

### Views worth building, in priority order

1. **UMAP scatter, coloured by cluster** — already built, upgrade with UMAP.
2. **Sankey per large cluster** — screen-to-screen flow. Shows where a journey
   actually breaks down, which a scatter never will.
3. **Pattern × cluster heatmap** (route B) — which patterns define which cluster.
   Directly interpretable, no projection artefacts.
4. **Anomaly overlay** — the same scatter with `markov_logprob` as a colour ramp.
   Anomalies should sit at cluster edges; if they sit in the middle, the
   thresholds need retuning.

---

## 6. Which route to use

**Ship route A, monitor with route B.**

Route A is the default in the fitted scorer: it is dense (every journey has a
representation), it degrades gracefully, and it needs no pattern table at
inference time. Route B leaves 9–21% of journeys with no evidence at all, which
is a liability in production.

Route B earns its place as the **interpretation and validation layer**:

- Its patterns are readable as-is — `HOME → payment → select bill → pay` is a
  sentence, an SVD component is not.
- `journey_candidates` gives a clustering-free archetype baseline.
- Cross-route ARI on non-noise journeys (0.51–0.58) is a structural validity
  check no single-route silhouette can provide.

**If the two routes ever stop agreeing on the core, something upstream broke.**
That is the cheapest regression test this pipeline has.

---

## Artefacts

| file | contents |
|---|---|
| `outputs/clusters/route_comparison.csv` | the §3 head-to-head table |
| `outputs/clusters/{platform}_{route}_catalog.csv` | one row per cluster: size, medoid path, behaviour stats |
| `outputs/clusters/{platform}_{route}_labels.csv` | journey → cluster + Markov log-prob + sequence |
| `outputs/clusters/{platform}_ngram_sweep.csv` | the n-gram evidence table |
| `outputs/clusters/{platform}_patterns_used.csv` | mined PrefixSpan patterns with support |
| `outputs/clusters/{platform}_scorer.pkl` | fitted scorer for inference |
| `outputs/clusters/{platform}_{route}_map.html` | interactive cluster map |
| `outputs/journeys/{platform}_journey_candidates.csv` | route B-2 clustering-free archetypes |
| `outputs/clusters/{platform}_{route}_catalog_named.csv` | catalog with human cluster names |

Human names for all 169 clusters, and what naming exposed about the settings, are
in [5_Cluster_Names.md](5_Cluster_Names.md).

## Reproducing

```bash
.venv/Scripts/python -m src.Rule_based.canonize
.venv/Scripts/python -m src.Rule_based.tokens
.venv/Scripts/python -m src.Rule_based.segment
.venv/Scripts/python -m src.Rule_based.postprocess
.venv/Scripts/python -m src.Rule_based.prefixspan
.venv/Scripts/python scripts/run_clustering.py --ngram-sweep
.venv/Scripts/python scripts/visualize_clusters.py --platform android
```
