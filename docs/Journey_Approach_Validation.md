# Journey Clustering — Direction Review and Working Pipeline

Reviewed against `data/final_clean_events.csv` (114,534 events · 1,425 sessions ·
182 devices · 154 customers · 2026-07-01 → 2026-07-07). Everything below is
measured on your data, not assumed.

---

## 1. Verdict on the direction

**The direction is right.** Canonize → tokenize → collapse repeats → cut into
journeys → cluster is the standard and correct pipeline for unsupervised
clickstream modelling, and it is the right foundation for the anomaly/friction
detection you want next. Two things in the plan needed to change, and one thing
needed adding.

### 1.1 The blocking problem: the field roles flip with `event_type`

This is the single most important finding, and it invalidates the tokenization
as currently specified.

| `event_type` | `segment_name` holds | `screen_name` holds |
| --- | --- | --- |
| `View` (86,690 rows) | **the screen** (`HomeVC`, `android/Home`, a webview URL) | NULL — **100.0%** missing |
| `Action` (27,844 rows) | **the action path** (`home/.../click_modem_control`) | **the screen** — 98.4% populated |

So `event_type + segment_name + screen_name` concatenates two *different*
meanings into one vocabulary. The practical damage:

- Every View token is `(View, screen, ∅)` and every Action token is
  `(Action, action_path, screen)`. The screen sits in a different tuple slot
  depending on the row, so **View and Action tokens live in disjoint
  namespaces**. Nothing in the model can learn that `View HomeVC` and
  `Action Home/Nav_profile` happen on the same screen.
- Without a shared screen axis there is no navigation graph, so "return to
  Home", "bounced between two screens", and screen-level transition features
  are all unavailable — and those are exactly the friction signals you are
  building towards.

**Fix.** Canonicalize to a role-correct triple before tokenizing:

```
View   ->  (View,   screen = segment_name,  target = ∅)
Action ->  (Action, screen = screen_name,   target = segment_name)
```

542 of the 611 Action screens then join cleanly onto the View screens. The
screen axis exists.

### 1.2 Your "don't merge `HOME` with `android/home`" rule — enforced mechanically

Agreed, and I made it structural rather than a matter of discipline: every
screen is namespaced by OS (`iOS::HomeVC`, `Android::HOME`). Identical strings
on different platforms **cannot** collide even by accident. No semantic merging
is performed anywhere in the pipeline.

The one canonicalization I do apply is purely structural, and it is the biggest
single win available:

- **575 of 821 distinct View screens are webview URLs**, and their long tail is
  almost entirely `contractNo=SGABN4765`, `orderId=8734`, `timestamp=...`
  variation — 11,046 rows carry a query string. Dropping non-semantic query
  params and masking id-like path segments (`/order/8734` → `/order/{id}`)
  removes *identifiers*, not behaviour. An allowlist (`tab`, `cat_id`,
  `orderType`, `step`, …) keeps params that genuinely change what the user sees.

Measured effect:

| stage | vocabulary | singletons | top-100 coverage |
| --- | --- | --- | --- |
| raw exact triple | 2,494 | 918 (36.8%) | 57.0% |
| canonical `(type, screen, target)` | 1,640 | 364 (22.2%) | 75.3% |
| canonical screen axis only | 423 | 20 (4.7%) | 91.5% |

### 1.3 The missing stage: `session_id` is not a session

Your plan treats journey segmentation as one step among many. On this data it is
**load-bearing**, because `session_id` does not delimit anything user-shaped:

- wall-clock span: median **3.2 min**, p75 18.9 min, **max 30.5 hours**
- events per session: median 34, max **1,524**
- inter-event gap: median **0.003 s** — events arrive in machine-fired bursts

Clustering whole sessions means clustering mixtures of unrelated goals. The
segmentation stage is what makes the unit of analysis meaningful.

---

## 2. What I built

```
src/journeylab/
  config.py       all tunables, one dataclass per stage
  canonize.py     role fix, URL canonicalization, OS namespacing, screen taxonomy
  tokens.py       multi-resolution tokens (L1/L2/L3) + rare-token backoff
  segment.py      rule-based + branching-entropy journey segmentation
  postprocess.py  consecutive-repeat, cycle, and chrome collapse
  features.py     n-gram TF-IDF + SVD + numeric block
  cluster.py      HDBSCAN / KMeans / per-cluster Markov chains
  score.py        inference path: new events -> cluster + anomaly + friction flags
scripts/run_journey_pipeline.py
```

Run it:

```bash
python scripts/run_journey_pipeline.py
```

Useful variants:

```bash
python scripts/run_journey_pipeline.py --level L3 --idle-gap 120 --entropy
```

---

## 3. Stage-by-stage: input, output, and what the data said

### Stage 2 — Tokenization (multi-resolution, not single)

One resolution cannot serve both modelling and interpretation, so three are
emitted per event:

| level | form | vocab | singletons | use |
| --- | --- | --- | --- | --- |
| **L1** | `V@iOS::HomeVC` | 636 | 7.2% | naming clusters, backoff target |
| **L2** | `A@iOS::HomeVC#Home/Nav_profile` | 1,259 | 15.5% | **modelling default** |
| **L3** | full action path | 1,640 | 22.2% | forensics, exact replay |

L3 is faithful but sparse — at L3, 22% of tokens appear exactly once, which is
noise to any distance metric. L2 truncates the action path to 2 components
(`home/home_service_management/*`), which is where the useful signal actually
lives.

**Rare-token backoff.** Document frequency is computed over *journeys*, not
events, on purpose: a token firing 200 times inside one journey is still one
piece of evidence. Tokens below `min_journey_df=3` fall back to their L1 form,
then to `<rare>`. In practice this touches only 1.2% of token instances —
the canonicalization already did the heavy lifting.

### Stage 3 — Journey segmentation

**Input:** canonical events sorted by `(session_id, ts)`.
**Output:** `journey_id`, `journey_pos`, `boundary_reason` per event.

Rule-based (`L0`, default) cuts on: session change · idle gap > τ · return to a
hub screen after ≥ k events · login/logout **action** · hard length cap.

Two rules were wrong on the first pass and the data caught both:

- `MainTabBarController` looks like the root screen but iOS pushes through it on
  essentially every navigation (7,049 events). Treating it as a hub made
  "return to root" fire 8,544 times and shredded sessions into 3-event
  fragments. **Hubs must be screens the user recognises as home, not the
  container hosting them** — it is excluded, with a comment saying why.
- Detecting auth transitions by matching `login` anywhere in the token fired on
  every `View LoginVC` during guest browsing. Auth is now detected on the
  **action target only** (`login/continue_login`, `Home/Account/Log_out`).

Idle-gap sensitivity (data: gap p97.5 = 73 s, p99 = 306 s):

| τ (s) | journeys | mean len | median len | share < 3 |
| --- | --- | --- | --- | --- |
| 30 | 12,314 | 9.30 | 7 | 13.8% |
| 60 | 10,824 | 10.58 | 7 | 10.2% |
| **90** | **10,262** | **11.16** | **7** | **8.9%** |
| 300 | 9,227 | 12.41 | 7 | 6.5% |
| 900 | 8,679 | 13.20 | 8 | 4.8% |

τ = 90 s is the recommended default: past ~90 s the journey count flattens, so
the threshold is in a stable region rather than on a cliff.

Final cut mix: root_return 4,987 · idle_gap 2,517 · session_start 1,425 ·
auth_change 1,264 · length_cap 69.

### Stage 4 — Post-tokenization cleanup

Your plan called for handling repeated consecutive sequences. There are actually
**three** kinds of redundancy here and only two of them are noise:

| kind | volume | treatment |
| --- | --- | --- |
| consecutive exact repeats `A A A` | 12,528 events | collapse — framework double-fire, pure noise |
| **cyclic repeats `A B A B A B`** | 2,731 events, 765 journeys | collapse **but keep the count** — this is a *friction signal*, not noise |
| OS chrome / boot screens | 31,758 events (27.7%) | drop from the sequence, keep as a count |

Cycle collapse is the piece your plan was missing. A user bouncing
`ModemSchedule ⇄ ManageModem` six times is the exact behaviour you want to
detect; collapsing it without recording it would delete your target variable.

**Nothing is discarded** — every removed event becomes a numeric feature
(`n_loop_removed`, `n_dedup_removed`, `n_chrome_events`, `max_run_length`).
Redundancy moves from the sequence channel to the numeric channel.

Net: 114,534 → 67,517 events (compression 0.59).

### Stage 5 — Representation

Two concatenated channels:

- **Sequence:** TF-IDF over 1–3-grams of the cleaned token sequence, with
  `<bos>`/`<eos>` sentinels so entry and exit points are first-class features,
  reduced by truncated SVD to 64 dims and L2-normalized. n-grams are what make
  order matter — `HOME→PAY→CONFIRM` and `HOME→SUPPORT→CHAT` share zero bigrams
  even when their unigram profiles overlap.
- **Numeric:** length, action ratio, back rate, loop count, revisit ratio,
  dwell and gap statistics — log1p'd (every one has a heavy right tail),
  standardized, and down-weighted to 0.35 so 13 scalars cannot outvote the
  sequence.

Result: 5,784 journeys × 77 dims, TF-IDF vocabulary 8,039.

> **On `duration`:** it is present on View rows only (Action duration is
> always 0), non-zero on 45.7% of Views, and its Spearman correlation with the
> gap to the next event is **−0.02** — so it is *not* a derived gap, it is the
> backend's own dwell measure. 54 rows exceed 1 hour and 6 exceed 24 hours;
> these are instrumentation artefacts and are clipped at 1,800 s rather than
> dropped.

---

## 4. Journey candidate algorithms

You asked which algorithms to use and what their input/output is. Splitting the
question in two, because "find journey boundaries" and "group journeys" are
different problems with different literature.

### 4.1 Finding journey candidates (segmentation)

| # | Algorithm | Input | Output | Verdict |
| --- | --- | --- | --- | --- |
| **A** | **Rule-based boundaries** (idle gap, hub return, auth, cap) | token stream + timestamps per session | boundary index list | **Implemented, use as baseline.** Interpretable, no fitting, product team can argue with it. |
| **B** | **Branching entropy** | token sequences | per-position H(next \| context); cut at high percentile | **Implemented (`--entropy`).** The principled unsupervised answer — inside a journey the next screen is near-determined; at a goal boundary the user can go anywhere. Use it to *audit* rule A: where they disagree, the rule list is probably missing a case. |
| **C** | **Sequitur / RePair grammar induction** | token sequences | hierarchical grammar; each rule = a recurring sub-journey | **Recommended next.** Gives "journey" a generative definition instead of a hand-tuned one, and the rule hierarchy is directly readable as a task tree. Best fit for discovering journeys you did not know to look for. |
| **D** | **Closed/maximal sequential pattern mining** (PrefixSpan, VMSP, CM-ClaSP) | set of sequences + `min_support` | frequent ordered subsequences with support | **Recommended, complementary.** Mines journey *templates* rather than cutting boundaries. Use it to build a named catalog ("package upgrade = these 6 steps in this order") and then match instances against it. Use *closed* patterns — plain frequent patterns explode combinatorially. |
| E | Voting Experts / MDL segmentation | token sequences | boundaries | Same family as B, more machinery. Only if B underperforms. |
| F | Change-point detection (PELT/BOCPD) on feature streams | per-event numeric features | change points | Weak here — the signal is categorical-sequential, not a continuous mean shift. |

**Recommendation:** ship A now (done), run B as a cross-check, add C or D as the
next iteration. A+D together is the highest-value pairing: A gives you units, D
gives you names for them.

### 4.2 Clustering journeys

| # | Algorithm | Input | Output | Verdict |
| --- | --- | --- | --- | --- |
| **1** | **n-gram TF-IDF → SVD → HDBSCAN** | journey token sequences | cluster label per journey, `-1` = noise | **Implemented, primary.** Scales, needs no `k`, and — decisive for your goal — *refuses* to assign genuinely odd journeys, giving you an anomaly signal for free. |
| **2** | **Per-cluster first-order Markov chains** | sequences + cluster labels | transition matrix per cluster; per-journey log-likelihood | **Implemented, primary companion.** Cluster membership tells you *which* archetype; only a generative model tells you *how badly* a journey fits it. This is what actually powers early error detection. |
| 3 | KMeans over the same embedding | embedding + `k` | labels | **Implemented as baseline only.** Forces every journey into a cluster, so it destroys the anomaly signal. Silhouette rises monotonically to k=30 (0.088 → 0.278) with no elbow — evidence that the structure here is density-shaped, not centroid-shaped. Use it to sanity-check, not to ship. |
| 4 | Edit-distance / LCS kernel → agglomerative | pairwise distance matrix | dendrogram + labels | Worth trying at this scale (5.8k journeys = 34M pairs, tractable). Captures alignment that n-grams approximate. Try if cluster 4/49-style long journeys look under-separated. |
| 5 | Mixture of Markov chains fitted by EM | sequences + `k` | soft assignments + generative model | The classic clickstream model. Elegant — clusters *and* likelihood in one fit — but needs more data than 5.8k journeys to fit a 871² transition matrix per component. Revisit after you have production volume. |
| 6 | Sequence autoencoder / doc2vec embeddings | sequences | dense vectors | **Not yet.** 5,784 journeys and 871 token types is well below where a learned embedding beats TF-IDF, and you would lose interpretability you currently have. |

---

## 5. Results on your data

HDBSCAN (`min_cluster_size=15`) over 5,784 journeys:

| metric | value |
| --- | --- |
| clusters | 57 |
| noise share | 33.1% |
| silhouette | 0.442 |
| Davies-Bouldin | 0.951 |
| cluster size | min 15 · median 50 · max 251 |
| stability (5× 80% subsample, ARI vs full fit) | 0.74 – 0.78 |

The clusters are nameable, which is the real test:

| cluster | n | what it is |
| --- | --- | --- |
| 46 / 49 | 184 / 87 | internet package upgrade via `dkol` webview (iOS / Android) |
| 24 | 251 | Android contract-list selection |
| 30 | 235 | guest taps banner → login popup |
| 20 | 148 | iOS OTP login |
| 47 | 160 | e-contract signing (PDF + identification) |
| 39 / 2 | 127 / 78 | spin-wheel game (iOS / Android) |
| 42 / 1 | 65 / 67 | notification browsing |
| 12 / 0 | 63 / 102 | modem restart scheduling |
| 4 | 63 | VNeID citizen-ID e-counter flow |

Friction ranking falls straight out — clusters ordered by back-rate and loop rate:

| cluster | n | back rate | mean loops | revisit ratio | median len | median span |
| --- | --- | --- | --- | --- | --- | --- |
| 4 (VNeID e-counter) | 63 | 0.173 | 0.48 | **0.369** | 22 | 89 s |
| 49 (Android pkg upgrade) | 87 | 0.145 | 0.28 | **0.500** | 29 | 86 s |
| 50 | 44 | **0.340** | 0.00 | 0.178 | 6.5 | 3 s |
| 39 (spin wheel) | 127 | 0.103 | **1.04** | 0.320 | 8 | 47 s |

Clusters 4 and 49 are the standouts: long, slow, half their screens revisited.
Those are your first two candidates for a UX review.

### The inference path works

`JourneyScorer` re-runs the identical pipeline on new events and returns cluster
assignment plus two **independent** anomaly channels — geometric (far from every
centroid: "unusual shape") and generative (low Markov log-prob: "unusual
transitions inside it"). They are kept separate rather than blended because they
disagree often enough to be diagnostically useful.

Smoke-tested on the final calendar day (4,443 events → 252 journeys):

```
matched a known archetype : 240 (95.2%)
geometric anomalies       :  12
generative anomalies      :  33
severe (both)             :   0
```

> Caveat: that day is inside the training window, so 95.2% is a check that the
> plumbing works, **not** a generalization estimate. A real evaluation needs a
> time-based split once you have more than one week of data.

---

## 6. What you should know before trusting the clusters

Stated plainly, because these limits are about the data, not the code:

1. **The sample is small and probably not production.** 154 customers, 182
   devices, 1,425 sessions over 7 days; 13,357 events on `staging-hi.fpt.vn`.
   The cluster catalog is a valid description of *this* traffic. Treating it as
   the production journey taxonomy, or fitting anomaly thresholds here and
   shipping them, would not be safe — QA traffic has different shape than real
   users. (Developer-localhost traffic is negligible: 13 journeys, 0.2%.)

2. **43% of journeys are dropped before clustering** (4,478 of 10,262) for
   falling below 4 tokens after cleanup. These are mostly trivial hops
   (`Home → tap → screen`) with no order information to cluster on. They are
   written to `journeys_all_including_short.csv` — but be aware the catalog
   describes the *substantial* journeys only.

3. **33% noise share is high**, and it is genuine: noise journeys are
   indistinguishable from clustered ones on length, back rate, and loop count
   (medians identical) — they are rare in *sequence*, which is exactly the thing
   HDBSCAN should refuse to force into a cluster. Lower `min_cluster_size` if
   you want more coverage, at the cost of less stable clusters.

4. **`screen_name` being 100% NULL on View rows is worth raising with the
   mobile team.** Everything above works around it, but if View events carried
   the screen in `screen_name` too, the role ambiguity would disappear at the
   source.

---

## 7. Suggested next steps

1. Get more data — ideally 4+ weeks of production traffic. Almost every caveat
   in §6 is a sample-size problem.
2. Run `--entropy` and diff its boundaries against the rule-based ones; each
   disagreement is either a missing rule or a mis-set threshold.
3. Add PrefixSpan/VMSP closed-pattern mining (§4.1 D) to name the catalog.
4. Have someone from the product team read the top 20 medoid paths in
   `outputs/journey/cluster_catalog.json` and label them. Human-named archetypes
   are what turn this from a clustering into a monitoring system.
5. Only then calibrate anomaly thresholds, on production data, with a
   time-based split.
