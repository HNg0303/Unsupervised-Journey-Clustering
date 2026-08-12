# Reusable prompt: name journey clusters consistently

Copy the prompt below when a new clustering run is fitted. Replace every
`<PLACEHOLDER>` before use.

---

## Prompt

You are naming customer-journey clusters for Hi FPT clickstream data.

### Inputs

- Canonical Vietnamese naming reference:
  `<REFERENCE_SHAREHOLDER_CLUSTER_CATALOG_VI_JSON>`
- Cluster catalog containing medoids and cluster metadata:
  `<PLATFORM_CLUSTER_CATALOG_JSON>`
- Ranked cluster n-grams:
  `<PLATFORM_CLUSTER_NGRAMS_CSV>`
- Platform: `<android|ios>`
- Model namespace/run: `<B|min_cluster_size_50|C|min_cluster_size_100|other>`

### Objective

Assign every non-noise cluster a readable Vietnamese `cluster_name` and
`business_family` that follow the vocabulary, tone, and granularity of the
reference catalog. Preserve an English audit name and all evidence used to
derive the name. Name noise as `Hành trình chưa phân loại / hỗn hợp`; never
force it into a business family.

### Evidence precedence

Evaluate evidence in this order:

1. The medoid's sequence of destination `view@...` surfaces.
2. Repeated transitions and destinations in the top three ranked n-grams.
3. Medoid entry and exit tokens.
4. Individual `action@...` tokens.
5. Behavioural metrics only as modifiers, such as back-out, looping, slow, or
   repeated navigation; metrics must not invent a business intent.

Use the medoid to establish the broad journey family. Use top-three n-grams to
confirm or refine the subtype. Lower-ranked n-grams may be retained for audit
but must not create a specific public label.

### Single-action guard

Do not name an entire cluster as a completed business journey when its evidence
contains only one unique action type and no matching destination view.

Examples:

- `Home → action@...SCAN_QR` without a QR destination view is a Home/navigation
  journey, not a completed `Quét mã QR` journey.
- `Home → action@...Nav_payment` without a payment view is not automatically a
  `Thanh toán` journey.
- `Home → action@...survey` followed only by a generic webview must use a
  coarser Home/navigation label unless several top n-grams independently
  confirm the survey destination.
- A specific label is allowed when the medoid or top-three n-grams contain a
  matching destination view, or when multiple independent n-grams consistently
  establish the same intent.

Record `single_action_guard_applied=true` and explain the suppressed action in
`naming_signals` whenever this rule changes the final name.

### Naming rules

1. Treat cluster IDs as arbitrary. Never infer meaning from ID proximity or
   reuse a previous name solely because an ID matches.
2. Prefer an exact Vietnamese display name already present in the reference
   catalog when the evidence represents the same intent.
3. Keep names concise and noun/intent oriented, for example:
   `Quản lý modem`, `Thông báo và cảnh báo`, or
   `Bảo vệ và quản lý thiết bị kết nối`.
4. Preserve the reference business-family vocabulary, such as:
   `quản lý thiết bị`, `tương tác`, `điều hướng`, `thanh toán`, `hỗ trợ`,
   `xác thực`, `hợp đồng`, `tài khoản`, `phản hồi`, `thương mại`,
   `tiện ích`, `quản lý dịch vụ`, `khác`, and `chưa phân loại`.
5. Do not overstate completion. Distinguish opening/navigation from submission,
   confirmation, payment completion, signing, blocking, or reset completion.
6. A subtype such as VietQR, prepaid, OTP, OAuth, 5 GHz, signing, or checkout
   requires support in the top three n-grams or a corresponding destination
   view. One incidental action is insufficient.
7. If several business intents compete and none dominates the medoid plus top
   evidence, choose the coarser surface/family name and lower confidence.
8. Use `high` confidence when medoid and n-grams agree, `medium` when only the
   broad surface is clear, and `not_applicable` for noise.
9. Preserve the original medoid path and n-grams verbatim for auditability.
10. Namespace output keys as `<NAMESPACE>:<cluster_id>`, for example `C:42` or
    `B:17`. Never merge raw B and C integer IDs.

### Required output per cluster

```json
{
  "namespace": "C",
  "cluster": 42,
  "effective_cluster_key": "C:42",
  "cluster_name": "Tên hiển thị tiếng Việt",
  "cluster_name_vi": "Tên hiển thị tiếng Việt",
  "cluster_name_en": "English audit name",
  "business_family": "nhóm nghiệp vụ tiếng Việt",
  "business_family_code": "stable_english_code",
  "naming_confidence": "high|medium|not_applicable",
  "naming_signals": ["bằng chứng ngắn gọn bằng tiếng Việt"],
  "naming_signals_en": ["English audit evidence"],
  "single_action_guard_applied": false,
  "medoid_journey_id": "...",
  "medoid_path": "...",
  "top_entry_token": "...",
  "top_exit_token": "...",
  "top_ngrams": []
}
```

### Validation before delivery

- Every non-noise cluster has exactly one namespaced key and one name.
- Noise remains unclassified.
- No B key is confused with a C key.
- Every specific subtype is supported by a destination view or top-three
  n-gram evidence.
- Every sole-action case has been audited using the single-action guard.
- Vietnamese names and families follow the canonical reference vocabulary.
- English audit names, raw medoids, and n-grams remain available.
- The inference mapping contains every fitted B and C cluster; missing keys
  safely fall back to `Hành trình chưa phân loại / hỗn hợp`.

Return the named JSON/CSV mappings, a list of clusters affected by the
single-action guard, and a short summary of name/family counts.

---

## Repository command

For the current B/C runs, the repository applies this policy with:

```powershell
python scripts/name_hierarchical_clusters.py `
  --postprocess-run output/journey_runs/L2_ng1-3_C-mcs100-ms5_B-mcs50-ms3_postprocessed `
  --reference-catalog output/clusters_with_screen/shareholder_cluster_catalog_vi.json
```

This writes named catalogs into the B run (`min_cluster_size=50`), the C run
(`min_cluster_size=100`), and the hierarchical post-processing folder used by
inference.
