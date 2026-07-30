# Cluster Names — Every Cluster, Both Platforms, Both Routes

Names for all 169 clusters (29 + 34 Android, 50 + 56 iOS, plus one noise group
per run), so the output can be discussed in words instead of integers.

Cluster ids are arbitrary — HDBSCAN assigns them by internal ordering and they
**move on every refit**. The names live in
[cluster_names.py](../src/Rule_based/cluster_names.py) and are applied with:

```bash
.venv/Scripts/python scripts/apply_cluster_names.py
```

Run `--audit` after any refit before quoting a name — it reports ids present in
the catalog but missing from the name table, and vice versa.

Each name came from three pieces of evidence: the **medoid path**, the
**highest-lift tokens** inside the cluster, and the **behavioural stats**
(length, back rate, revisit ratio, loop share). The evidence is kept in the
`note` field so any name can be challenged rather than taken on trust.

---

## 1. Families — the level to actually report on

169 cluster names is too many for a conversation. Every cluster carries a
`family`, and that is the unit to put in a slide.

### Android

| family | route A clusters | route A journeys | route B clusters | route B journeys |
|---|---|---|---|---|
| payment | 3 | 264 (14.7%) | 5 | 287 (16.0%) |
| contract | 2 | 158 (8.8%) | 7 | 182 (10.1%) |
| auth | 8 | 209 (11.6%) | 5 | 221 (12.3%) |
| service | 2 | 119 (6.6%) | 2 | 57 (3.2%) |
| support | 3 | 92 (5.1%) | 3 | 112 (6.2%) |
| econtract | 2 | 66 (3.7%) | 2 | 78 (4.4%) |
| account | 3 | 70 (3.9%) | 4 | 86 (4.8%) |
| ecounter | 2 | 46 (2.6%) | 1 | 31 (1.7%) |
| guest | 1 | 21 (1.2%) | 2 | 32 (1.8%) |
| utility | 1 | 21 (1.2%) | 1 | 29 (1.6%) |
| promotion | 1 | 18 (1.0%) | — | — |
| notification | 1 | 17 (1.0%) | — | — |
| navigation | — | — | 2 | 42 (2.3%) |
| **noise** | 1 | **694 (38.7%)** | 1 | **638 (35.5%)** |

### iOS

| family | route A clusters | route A journeys | route B clusters | route B journeys |
|---|---|---|---|---|
| service | 4 | 365 (9.0%) | 12 | 398 (9.8%) |
| auth | 9 | 357 (8.8%) | 3 | 244 (6.0%) |
| contract | 10 | 324 (8.0%) | 14 | 430 (10.6%) |
| payment | 5 | 226 (5.6%) | 5 | 273 (6.7%) |
| shop | 7 | 222 (5.5%) | 3 | 128 (3.2%) |
| guest | 1 | 195 (4.8%) | 2 | 60 (1.5%) |
| econtract | 4 | 184 (4.5%) | 3 | 160 (3.9%) |
| support | 1 | 167 (4.1%) | 2 | 95 (2.3%) |
| popup | 2 | 144 (3.5%) | 3 | 127 (3.1%) |
| notification | 2 | 119 (2.9%) | — | — |
| account | 3 | 114 (2.8%) | 7 | 225 (5.5%) |
| promotion | 1 | 38 (0.9%) | — | — |
| ecounter | 1 | 20 (0.5%) | — | — |
| navigation | — | — | 2 | 153 (3.8%) |
| **noise** | 1 | **1,588 (39.1%)** | 1 | **1,770 (43.6%)** |

**Platform difference in one line:** Android's biggest named family is
**payment** (14.7%); iOS's is **service** (9.0%) with **auth** and **contract**
right behind. Android users in this window came to pay bills; iOS users came to
manage their internet package and switch contracts.

`shop` (5.5%) and `notification` (2.9%) exist on iOS route A and have **no
Android counterpart at all** — those flows are webview-driven on iOS and either
absent or too rare to cluster on Android.

---

## 2. What naming exposed

Naming every cluster surfaced three things that the quality metrics could not.

### 2.1 iOS over-fragments logout and contract switching

Route A found **9 auth clusters on iOS — 7 of them are logout variants**:

| id | name | n | what distinguishes it |
|---|---|---|---|
| 15 | Logout then clear login form | 44 | `login_button_clear` after |
| 14 | Logout | 24 | plain |
| 19 | Logout (fast) | 22 | 4 steps, action ratio 0.50 |
| 16 | Logout (clean) | 19 | 4 steps, revisit 0.0 |
| 13 | Logout via account popup | 19 | popup path |
| 3 | Logout from personal info | 15 | starts on PersonalVC |
| 20 | Logout with form clear | 15 | `login_button_clear` |

That is **158 journeys split 7 ways over differences no one would report**.
Same story for `contract`: 10 route-A clusters (14 on route B) that are almost
all "user switched contract", separated by whether they backed out, confirmed,
or continued into a service screen.

**This is a real finding about the settings, not about users.** For iOS,
`min_cluster_size = 15` is too small relative to 4,063 journeys. Two options:

- report at **family** level (the tables in §1) and treat cluster ids as detail; or
- refit iOS with `min_cluster_size` 30–40 and expect ~25–30 clusters instead of 50.

Android does not have this problem to the same degree — 1,795 journeys over 29
clusters is a reasonable ratio — though it has its own near-duplicates (below).

### 2.2 Near-duplicate pairs worth merging

Clusters that describe the same behaviour and differ only in length or one token:

| platform | route | pair | shared behaviour |
|---|---|---|---|
| Android | A | **24** + **10** | support request opened then closed (back 0.18 / 0.19) |
| Android | A | **27** + **28** | logout; 28 just lands on guest home after |
| Android | A | **21** + **22** | guest login header tap; 21 dismisses an invite popup first |
| Android | A | **3** + **4** | OTP login, guest vs member entry path |
| Android | B | **14** + **11** | guest login prompt, identical shape |
| Android | B | **34** + **33** | support request closed |
| iOS | B | **28** + **3** | reminder popup skipped |
| iOS | B | **54/43/33/56/55** | contract switch, five ways |

These are genuine — the sequences really do differ — but not *usefully* genuine.
The family roll-up already merges them; this list is what to check first if you
tune `min_cluster_size`.

### 2.3 Friction ranked, by name

The clusters to actually act on. Ranked by revisit ratio and loop share taken
together, not by either alone — a cluster can loop without thrashing and vice
versa:

| platform | id | name | n | len | revisit | loops |
|---|---|---|---|---|---|---|
| iOS | 40 | **E-contract list thrash** | 17 | 38 | 0.64 | 53% |
| iOS | 49 | **Package/product webview thrash** | 28 | 18 | 0.57 | 68% |
| iOS | 41 | **Pay on behalf** | 27 | 20 | 0.52 | 56% |
| iOS | 39 | **E-contract identity input** | 81 | 17 | 0.39 | **52%** |
| iOS | 48 | **Shop webview browse** | 55 | 6 | 0.38 | 51% |
| Android | 13 | **Change address (e-counter)** | 15 | 34 | 0.49 | 47% |
| iOS | 37 | **Contract sharing and permissions** | 68 | 23 | 0.50 | 32% |
| iOS | 47 | **Internet package upgrade (web)** | 188 | 21 | 0.44 | 20% |
| Android | 19 | **Internet package upgrade (web)** | 90 | 21 | 0.41 | 26% |
| Android | 12 | **Change service location** | 31 | 16 | 0.44 | 6% |

Two patterns:

1. **Webview flows dominate.** `update-package`, `web/shop`, `product-detail`
   and the e-contract PDF host are in almost every high-friction cluster on both
   platforms. The native screens are comparatively clean.
2. **iOS 39 "E-contract identity input" is the one to escalate** — 81 journeys,
   over half containing a detected navigation loop. It is the largest cluster in
   the high-friction group by a factor of three.

Counterpoint, so this reads fairly: the cleanest clusters are the auth ones.
Android 1 "OAuth login" and iOS 10 "OTP login" both have **zero back rate and
near-zero revisit**. Login is not where the problem is.

---

## 3. Full naming tables

Columns: `n` = journeys, `len` = median cleaned length, `back` = mean back rate,
`revisit` = mean revisit ratio, `loops` = share of journeys with a detected
navigation loop.

### android — route A (tf-idf)

| family | id | name | n | len | back | revisit | loops | evidence |
|---|---|---|---|---|---|---|---|---|
| payment | 25 | **Guest bill payment** | 129 | 11 | 0.05 | 0.21 | 16% | full guest flow: billing tab > search contract > select > pay |
| payment | 26 | **Postpaid bill payment** | 75 | 8 | 0.05 | 0.22 | 11% | payment tab > select bill > click pay > payment info |
| payment | 18 | **Prepaid / utilities browse** | 60 | 7 | 0.09 | 0.24 | 8% | enters prepaid or utilities, backs out. 7s median |
| contract | 8 | **Contract switching** | 121 | 7 | 0.12 | 0.19 | 2% | change_contract > contract list > choose > back. Fast and mechanical |
| contract | 15 | **Contract sharing and permissions** | 37 | 20 | 0.07 | 0.40 | 8% | e-counter share management: add phone, linked phone, permissions |
| econtract | 20 | **E-contract signing with VNeID** | 47 | 24 | 0.10 | 0.38 | 19% | unconfirmed e-contract > VNeID > confirm CCCD. 24 steps |
| econtract | 17 | **E-contract peek and back out** | 19 | 6 | 0.11 | 0.30 | 5% | opens unconfirmed e-contract, hits hd_pl/back |
| service | 19 | **Internet package upgrade (web)** | 90 | 21 | 0.16 | 0.41 | 26% | dkol/update-package webview. revisit 0.41, loops 26% - friction |
| service | 9 | **Modem control** | 29 | 14 | 0.08 | 0.28 | 7% | home service management > internet tab > modem control |
| auth | 1 | **OAuth login (external)** | 44 | 4 | 0.00 | 0.00 | 0% | login > AuthorizationManagement > RedirectUriReceiver. Zero back, zero revisit |
| auth | 3 | **OTP login (guest path)** | 36 | 5 | 0.01 | 0.12 | 0% | guest login > OTP screen |
| auth | 21 | **Guest login prompt (after invite popup)** | 34 | 6 | 0.10 | 0.11 | 3% | dismisses invite popup, then taps login header |
| auth | 27 | **Logout** | 24 | 4 | 0.00 | 0.01 | 0% | Account > Log_out > confirm popup > login screen |
| auth | 28 | **Logout then guest browse** | 20 | 5 | 0.00 | 0.09 | 5% | logout, lands on guest home instead of leaving |
| auth | 4 | **OTP login (member path)** | 19 | 6 | 0.00 | 0.12 | 0% | continue_login > login_with_otp > OTP screen |
| auth | 22 | **Guest login tap (minimal)** | 17 | 4 | 0.03 | 0.04 | 6% | guest home > login header > login. 4 steps, no popup |
| auth | 2 | **FID login** | 15 | 5 | 0.00 | 0.00 | 0% | continue_login > login_with_fid > Authorization |
| account | 11 | **Account > e-contract browse** | 40 | 6 | 0.07 | 0.26 | 5% | account home > acceptance record / e-contract list |
| account | 7 | **Profile open** | 15 | 5 | 0.00 | 0.23 | 0% | Nav_profile > account home. 5 steps |
| account | 14 | **Profile > contract management** | 15 | 5 | 0.02 | 0.12 | 20% | Nav_profile > account > contract management |
| support | 24 | **Support request opened then closed** | 44 | 6 | 0.18 | 0.10 | 2% | Nav_support > support_create > confirm_close. back 0.18 |
| support | 23 | **Support request submit (fast)** | 31 | 5 | 0.06 | 0.26 | 6% | 0.9s median span - submits without reading |
| support | 10 | **Support request closed immediately** | 17 | 5 | 0.19 | 0.01 | 0% | same shape as A24, shorter. back 0.19 |
| notification | 0 | **Notification detail** | 17 | 7 | 0.02 | 0.31 | 41% | notification detail activity. loops in 41% of journeys |
| promotion | 5 | **Promotions browse** | 18 | 8 | 0.12 | 0.23 | 6% | Nav_promotion > loyalty promotion webview |
| guest | 16 | **Guest payment hits login wall** | 21 | 9 | 0.06 | 0.25 | 0% | guest billing > LoginForGuestDialog. Conversion blocker |
| ecounter | 12 | **Change service location** | 31 | 16 | 0.11 | 0.44 | 6% | e-counter change location > verify |
| ecounter | 13 | **Change address (e-counter)** | 15 | 34 | 0.06 | 0.49 | 47% | 34 steps, revisit 0.49, loops 47% - worst Android friction |
| utility | 6 | **QR scan** | 21 | 6 | 0.05 | 0.23 | 5% | home header > scan_qr screen |
| noise | -1 | *Unassigned* | 694 | 7 | 0.09 | 0.19 | 8% | no dense neighbourhood |

### android — route B (PrefixSpan)

| family | id | name | n | len | back | revisit | loops | evidence |
|---|---|---|---|---|---|---|---|---|
| payment | 3 | **Guest bill payment** | 94 | 12 | 0.04 | 0.20 | 13% | guest billing checkbox + buttons + text field |
| payment | 6 | **Bill payment** | 77 | 7 | 0.09 | 0.24 | 6% | payment screen > payment info |
| payment | 20 | **Postpaid bill payment** | 68 | 10 | 0.06 | 0.24 | 19% | select bill > click pay |
| payment | 7 | **Payment tab bounce** | 31 | 8 | 0.06 | 0.23 | 13% | Nav_payement > payment > back to home |
| payment | 19 | **Payment after invite dismiss** | 17 | 5 | 0.09 | 0.07 | 0% | invite_update_close then Nav_payement |
| contract | 24 | **Contract sharing** | 51 | 13 | 0.05 | 0.28 | 18% | contract management > e-counter share management |
| contract | 29 | **Contract switch then service** | 46 | 10 | 0.14 | 0.35 | 15% | change_contract + service management |
| contract | 26 | **Contract management and list** | 21 | 23 | 0.11 | 0.45 | 14% | 23 steps, revisit 0.45 |
| contract | 30 | **Contract switch** | 17 | 7 | 0.14 | 0.17 | 0% | change_contract > contract list |
| contract | 25 | **Contract sharing (long)** | 16 | 34 | 0.09 | 0.52 | 19% | 33.5 steps, revisit 0.52 |
| contract | 28 | **Contract switch abandoned** | 16 | 8 | 0.14 | 0.20 | 0% | change_contract > choose > btn_back |
| contract | 16 | **Contract switch (short)** | 15 | 7 | 0.08 | 0.20 | 0% | change_contract > contract list, 7 steps |
| econtract | 32 | **E-contract signing (VNeID)** | 52 | 8 | 0.09 | 0.32 | 8% | go_to_screen > unconfirmed e-contract > confirm CCCD |
| econtract | 18 | **E-contract confirm (deep)** | 26 | 18 | 0.13 | 0.38 | 19% | deep FE_CONTRACT_BLOCK / CONTRACT_CONFIRM path |
| service | 1 | **Internet package upgrade (web)** | 37 | 30 | 0.18 | 0.59 | 43% | 30 steps, revisit 0.59, loops 43% - friction |
| service | 21 | **Modem control** | 20 | 12 | 0.08 | 0.27 | 5% | internet tab > modem control |
| auth | 14 | **Guest login prompt** | 68 | 5 | 0.03 | 0.07 | 9% | guest home > login header > login |
| auth | 4 | **OAuth login (external)** | 41 | 4 | 0.00 | 0.00 | 0% | Authorization > RedirectUriReceiver |
| auth | 11 | **Guest login prompt (variant)** | 40 | 6 | 0.10 | 0.23 | 0% | same shape as B14 |
| auth | 12 | **OTP login** | 37 | 6 | 0.01 | 0.11 | 0% | guest login > OTP screen |
| auth | 5 | **FID login** | 35 | 5 | 0.00 | 0.05 | 3% | login_with_fid > Authorization |
| account | 23 | **Profile open** | 35 | 6 | 0.04 | 0.17 | 6% | Nav_profile > account |
| account | 2 | **Profile then payment** | 17 | 7 | 0.05 | 0.22 | 0% | Nav_profile > account > payment > login |
| account | 13 | **Profile / home toggle** | 17 | 6 | 0.03 | 0.24 | 0% | Nav_profile and Nav_home alternating |
| account | 15 | **Account > e-contract** | 17 | 5 | 0.10 | 0.09 | 0% | account > contract management > e-contract |
| support | 17 | **Support request** | 54 | 6 | 0.11 | 0.21 | 4% | support_create + return to home |
| support | 34 | **Support request closed** | 38 | 5 | 0.19 | 0.03 | 5% | Nav_support > confirm_close |
| support | 33 | **Support request closed (variant)** | 20 | 7 | 0.16 | 0.19 | 0% | same shape as B34 |
| guest | 10 | **Guest payment login wall** | 17 | 9 | 0.04 | 0.20 | 6% | guest billing > LoginForGuestDialog |
| guest | 9 | **Guest bill lookup** | 15 | 8 | 0.08 | 0.21 | 0% | invite dismiss > guest billing text field |
| ecounter | 8 | **Change service location** | 31 | 9 | 0.11 | 0.31 | 3% | contract management > e-counter change location |
| utility | 22 | **QR scan** | 29 | 7 | 0.03 | 0.22 | 14% | home header > scan_qr |
| navigation | 31 | **Home browse** | 27 | 5 | 0.16 | 0.05 | 4% | HOME + android/Home only, no destination |
| navigation | 27 | **Home browse (short)** | 15 | 4 | 0.11 | 0.26 | 0% | 4 steps on HOME only |
| noise | -1 | *Unassigned* | 638 | 6 | 0.10 | 0.19 | 9% | no dense neighbourhood |

### ios — route A (tf-idf)

| family | id | name | n | len | back | revisit | loops | evidence |
|---|---|---|---|---|---|---|---|---|
| payment | 31 | **Payment tab bounce** | 82 | 6 | 0.01 | 0.23 | 2% | Nav_payement > PaymentHome > back to home |
| payment | 45 | **Bill payment (full flow)** | 48 | 14 | 0.01 | 0.38 | 15% | select_bill > pay > PayAction > PaymentResult |
| payment | 27 | **Payment settings and schedule** | 37 | 18 | 0.03 | 0.44 | 19% | Payment/Setting > PaymentSchedule / ExtendService |
| payment | 34 | **Prepaid payment** | 32 | 13 | 0.01 | 0.41 | 16% | choose_prepaid > PrePaidVC > pay |
| payment | 41 | **Pay on behalf** | 27 | 20 | 0.00 | 0.52 | 56% | BehalfPayment > BehalfPaymentResult. loops 56% |
| contract | 37 | **Contract sharing and permissions** | 68 | 22 | 0.03 | 0.50 | 32% | share management, add/remove linked phone. revisit 0.50 |
| contract | 22 | **Contract pick > e-contract** | 40 | 5 | 0.00 | 0.16 | 2% | choose_contract > EContractHome |
| contract | 23 | **Contract switch** | 32 | 4 | 0.00 | 0.21 | 0% | Home/change_contract > ChooseContract |
| contract | 6 | **Contract pick > service** | 30 | 5 | 0.00 | 0.11 | 0% | choose_contract > ServiceManage |
| contract | 18 | **Contract switch (fast)** | 30 | 4 | 0.02 | 0.05 | 0% | change_contract > choose_contract. 4 steps |
| contract | 4 | **Contract switch (with return)** | 29 | 5 | 0.04 | 0.23 | 0% | change_contract > choose, returns to Home |
| contract | 24 | **Contract switch abandoned** | 27 | 5 | 0.17 | 0.14 | 0% | change_contract > btn_back / list reload. back 0.17 |
| contract | 21 | **Contract list back-out** | 25 | 4 | 0.23 | 0.20 | 0% | ChooseContract > btn_back. back 0.23 |
| contract | 17 | **Contract switch (confirmed)** | 23 | 5 | 0.00 | 0.20 | 0% | change_contract > choose_contract, both present |
| contract | 5 | **Contract switch > service** | 20 | 6 | 0.00 | 0.23 | 0% | change_contract > choose > ServiceManage |
| econtract | 39 | **E-contract identity input** | 81 | 17 | 0.01 | 0.39 | 52% | tab_info_order > InputIdentification. loops in 52% |
| econtract | 33 | **E-contract peek and back out** | 58 | 5 | 0.16 | 0.23 | 5% | EContractHome > hd_pl/back |
| econtract | 36 | **E-contract PDF sign** | 28 | 12 | 0.01 | 0.30 | 18% | tab_hd_plhd > PDFHost > SuccessSignEcontract |
| econtract | 40 | **E-contract list thrash** | 17 | 38 | 0.08 | 0.64 | 53% | 38 steps, revisit 0.64, loops 53% - worst friction on iOS |
| service | 47 | **Internet package upgrade (web)** | 188 | 21 | 0.04 | 0.44 | 20% | dkol/update-package. revisit 0.44, loops 20% - friction |
| service | 32 | **Service manage back-out** | 138 | 5 | 0.20 | 0.12 | 0% | Home > btn_back > ServiceManage. back 0.20 |
| service | 7 | **Contract pick > internet manage** | 24 | 6 | 0.01 | 0.12 | 0% | choose_contract > other_manage_internet > ServiceManage |
| service | 35 | **Package page immediate back-out** | 15 | 4 | 0.23 | 0.16 | 7% | update-package > webHeader BackButton. back 0.23 |
| auth | 10 | **OTP login** | 159 | 5 | 0.01 | 0.17 | 1% | continue_login > login_with_otp > OTPCreatePin. Cleanest cluster in the set |
| auth | 15 | **Logout then clear login form** | 44 | 5 | 0.00 | 0.00 | 0% | Log_out > popup > LoginVC login_button_clear |
| auth | 9 | **OTP login > contract pick** | 40 | 5 | 0.00 | 0.00 | 0% | OTP login then ChooseContract |
| auth | 14 | **Logout** | 24 | 5 | 0.00 | 0.20 | 0% | Log_out > popup > LoginVC |
| auth | 19 | **Logout (fast)** | 22 | 4 | 0.00 | 0.01 | 0% | 4 steps, action ratio 0.50 |
| auth | 13 | **Logout via account popup** | 19 | 4 | 0.00 | 0.03 | 0% | AccountHome popup > LoginVC login_button_clear |
| auth | 16 | **Logout (clean)** | 19 | 4 | 0.00 | 0.00 | 0% | 4 steps, revisit 0.0 |
| auth | 3 | **Logout from personal info** | 15 | 5 | 0.01 | 0.01 | 0% | PersonalVC popup > Log_out > LoginVC |
| auth | 20 | **Logout with form clear** | 15 | 5 | 0.00 | 0.00 | 0% | Log_out > login_button_clear |
| account | 11 | **Profile open and return** | 44 | 4 | 0.02 | 0.22 | 0% | Nav_profile > AccountHome > Home |
| account | 12 | **Profile open and return (tab)** | 37 | 5 | 0.01 | 0.23 | 0% | Nav_profile > AccountHome > Nav_home |
| account | 2 | **Personal info screen** | 33 | 6 | 0.03 | 0.23 | 0% | Nav_profile > PersonalVC > Nav_home |
| support | 8 | **Support article browse** | 167 | 7 | 0.03 | 0.26 | 12% | support page view controller > description > request list |
| shop | 48 | **Shop webview browse** | 55 | 6 | 0.01 | 0.38 | 51% | STWebRemote > web/shop/home. loops 51% |
| shop | 43 | **Shop order history** | 39 | 12 | 0.24 | 0.28 | 18% | orderHistoryButton > order-history > OrderDetail. back 0.24 |
| shop | 42 | **Shop webview back-out** | 33 | 6 | 0.24 | 0.23 | 3% | web/shop/home > BackButton. back 0.24 |
| shop | 49 | **Package/product webview thrash** | 28 | 18 | 0.03 | 0.57 | 68% | revisit 0.57, loops 68% - worst loop share on iOS |
| shop | 44 | **Product detail webview** | 25 | 11 | 0.02 | 0.50 | 40% | dkol/product-detail > CreateOrder. revisit 0.50 |
| shop | 0 | **Shop search** | 24 | 5 | 0.03 | 0.23 | 4% | open_url_in_app_with_access_token > web/shop/search |
| shop | 46 | **Package order and payment (web)** | 18 | 26 | 0.06 | 0.32 | 28% | order-history > UpdatePackage policy > payment_infor. 25 steps |
| notification | 30 | **Notification browse > detail** | 76 | 9 | 0.01 | 0.36 | 34% | view_all > view_noti > DetailsNoti. loops 34% |
| notification | 29 | **Notification list open** | 43 | 5 | 0.01 | 0.13 | 0% | Header go_to_screen > noti/view_all |
| promotion | 1 | **Promotions browse** | 38 | 4 | 0.01 | 0.18 | 13% | Nav_promotion > loyalty promotion webview |
| popup | 28 | **Reminder popup skipped** | 109 | 5 | 0.02 | 0.24 | 9% | remind_noti shown > skip_remind |
| popup | 25 | **Invite popup dismissed** | 35 | 4 | 0.15 | 0.19 | 3% | PopupBigMessage > invite_update_close |
| guest | 26 | **Guest banner tap > login wall** | 195 | 5 | 0.03 | 0.18 | 4% | guest banner > PopupInviteLogin. Largest iOS archetype |
| ecounter | 38 | **Change contact info (e-counter)** | 20 | 27 | 0.08 | 0.46 | 40% | change phone/email > confirm. 27 steps, loops 40% |
| noise | -1 | *Unassigned* | 1588 | 6 | 0.05 | 0.23 | 14% | no dense neighbourhood |

### ios — route B (PrefixSpan)

| family | id | name | n | len | back | revisit | loops | evidence |
|---|---|---|---|---|---|---|---|---|
| payment | 47 | **Payment tab > bill info** | 107 | 9 | 0.02 | 0.34 | 6% | Nav_payement > PaymentHome > payment_infor |
| payment | 50 | **Bill payment (full)** | 87 | 18 | 0.03 | 0.45 | 31% | payment_infor > PayAction > PaymentResult. 18 steps |
| payment | 34 | **Prepaid payment** | 28 | 20 | 0.01 | 0.48 | 25% | choose_contract > PrePaidVC > Nav_payement. 20 steps |
| payment | 2 | **Payment home browse** | 27 | 5 | 0.03 | 0.07 | 7% | PaymentHome > HomeVC only |
| payment | 19 | **Payment tab open** | 24 | 5 | 0.01 | 0.04 | 0% | Nav_payement > PaymentHome |
| contract | 54 | **Contract switch** | 57 | 5 | 0.00 | 0.22 | 0% | change_contract > choose_contract |
| contract | 43 | **Contract switch (fast)** | 52 | 4 | 0.06 | 0.02 | 0% | 4 steps |
| contract | 39 | **Contract switch (no pick)** | 41 | 5 | 0.07 | 0.02 | 0% | change_contract but never chooses |
| contract | 33 | **Contract switch (variant)** | 38 | 5 | 0.00 | 0.17 | 0% | choose_contract first, change_contract after |
| contract | 48 | **Contract switch abandoned** | 37 | 5 | 0.07 | 0.26 | 3% | change_contract > btn_back |
| contract | 52 | **Contract sharing** | 37 | 13 | 0.01 | 0.42 | 30% | ManagerContract > ShareManagement. 13 steps |
| contract | 46 | **Contract list browse** | 34 | 5 | 0.08 | 0.25 | 6% | ChooseContract without acting |
| contract | 31 | **Contract switch abandoned (variant)** | 26 | 5 | 0.14 | 0.21 | 0% | ChooseContract btn_back |
| contract | 40 | **Contract switch back-out** | 22 | 5 | 0.18 | 0.00 | 0% | btn_back > change_contract > ServiceManage |
| contract | 35 | **Contract sharing and permissions** | 21 | 37 | 0.03 | 0.55 | 38% | ManagerContract + PhonePermissions. 37 steps, revisit 0.55 |
| contract | 12 | **Contract management** | 19 | 16 | 0.02 | 0.52 | 37% | ManagerContract go_to_screen. 16 steps |
| contract | 56 | **Contract switch (variant 2)** | 16 | 5 | 0.00 | 0.24 | 0% | change_contract > choose_contract |
| contract | 38 | **Contract switch abandoned (short)** | 15 | 5 | 0.07 | 0.23 | 0% | change_contract > btn_back |
| contract | 55 | **Contract switch (variant 3)** | 15 | 6 | 0.00 | 0.35 | 0% | change_contract > choose_contract |
| econtract | 45 | **E-contract PDF view** | 78 | 8 | 0.07 | 0.24 | 26% | PDFHost + loading view + hd_pl tabs |
| econtract | 26 | **E-contract peek and back** | 63 | 6 | 0.09 | 0.29 | 11% | hd_pl/back + tab switching |
| econtract | 49 | **E-contract payment** | 19 | 32 | 0.02 | 0.53 | 74% | EContract > other_payment_method > PayAction. loops 74% |
| service | 29 | **Contract pick > internet manage** | 58 | 5 | 0.03 | 0.11 | 0% | choose_contract > other_manage_internet |
| service | 10 | **Package upgrade > order** | 53 | 37 | 0.04 | 0.53 | 45% | update-package > CreateOrder. 37 steps, loops 45% |
| service | 23 | **Internet manage open** | 49 | 5 | 0.07 | 0.15 | 0% | other_manage_internet > ServiceManage |
| service | 6 | **Package upgrade browse** | 41 | 32 | 0.05 | 0.53 | 34% | update-package + BackButton. 32 steps |
| service | 24 | **Service manage back-out** | 37 | 4 | 0.21 | 0.01 | 3% | Home btn_back > ServiceManage. back 0.21 |
| service | 22 | **Internet manage back-out** | 29 | 5 | 0.19 | 0.21 | 0% | btn_back > other_manage_internet. back 0.19 |
| service | 20 | **Package page back-out** | 28 | 5 | 0.19 | 0.23 | 14% | update-package > BackButton. back 0.19 |
| service | 4 | **Package order confirm** | 25 | 15 | 0.04 | 0.22 | 28% | UpdatePackage policy > continue > CreateOrder |
| service | 17 | **Modem and WiFi management** | 24 | 24 | 0.08 | 0.47 | 25% | ManageModem + ManageWiFi + connected devices. 23.5 steps |
| service | 5 | **TV service package browse** | 23 | 34 | 0.06 | 0.55 | 9% | click_tv_service + update-package. 34 steps, revisit 0.55 |
| service | 18 | **Package > contract switch** | 16 | 10 | 0.13 | 0.38 | 25% | ServiceManage change_contract > BackButton |
| service | 11 | **Package order (short)** | 15 | 19 | 0.05 | 0.35 | 27% | UpdatePackage continue > CreateOrder. 19 steps |
| auth | 32 | **OTP login** | 188 | 5 | 0.01 | 0.13 | 1% | continue_login > login_with_otp > OTPCreatePin |
| auth | 1 | **Logout then re-login** | 31 | 5 | 0.04 | 0.12 | 6% | login_button_clear > OTPCreatePin |
| auth | 21 | **FID login** | 25 | 4 | 0.01 | 0.09 | 0% | login_with_fid > continue_login > OTP |
| account | 36 | **Profile open** | 59 | 5 | 0.07 | 0.09 | 2% | Nav_profile > AccountHome |
| account | 37 | **Profile / home toggle** | 38 | 6 | 0.03 | 0.29 | 3% | AccountHome > Nav_home alternating |
| account | 42 | **Profile revisit loop** | 34 | 9 | 0.04 | 0.33 | 26% | Nav_profile > AccountHome, loops 26% |
| account | 53 | **Profile open (fast)** | 29 | 4 | 0.00 | 0.23 | 0% | 4 steps |
| account | 13 | **Account home browse** | 27 | 5 | 0.03 | 0.16 | 4% | AccountHome > HomeVC only |
| account | 8 | **Personal info screen** | 19 | 6 | 0.01 | 0.27 | 0% | PersonalVC > Nav_home |
| account | 30 | **Profile open (minimal)** | 19 | 4 | 0.06 | 0.01 | 0% | 4 steps |
| support | 14 | **Support article browse** | 69 | 6 | 0.03 | 0.22 | 12% | support page view controllers |
| support | 27 | **Support browse > e-contract** | 26 | 8 | 0.04 | 0.26 | 0% | support pages then EContractHome |
| shop | 16 | **Shop webview browse** | 95 | 7 | 0.03 | 0.38 | 47% | STWebRemote / STWebpage. loops 47% |
| shop | 44 | **Shop order history (web)** | 18 | 26 | 0.04 | 0.64 | 72% | order-history > OrderDetail > product-detail. loops 72% |
| shop | 51 | **Product / order webview** | 15 | 9 | 0.05 | 0.50 | 60% | product-detail + order-detail. loops 60% |
| popup | 28 | **Reminder popup skipped** | 68 | 5 | 0.00 | 0.26 | 7% | remind_noti > skip_remind |
| popup | 15 | **Popup sequence** | 30 | 9 | 0.04 | 0.41 | 37% | PopupVC chain. loops 37% |
| popup | 3 | **Reminder popup skipped (variant)** | 29 | 5 | 0.04 | 0.03 | 10% | same shape as B28 |
| guest | 9 | **Guest profile > contract** | 40 | 8 | 0.05 | 0.37 | 22% | guest Nav_profile > AccountHome > ManagerContract |
| guest | 25 | **Guest payment tab bounce** | 20 | 8 | 0.02 | 0.30 | 5% | guest home > PaymentHome > Nav_home |
| navigation | 7 | **Home browse** | 130 | 5 | 0.04 | 0.29 | 6% | HomeVC only, no destination reached |
| navigation | 41 | **Home deeplink > payment / e-contract** | 23 | 9 | 0.04 | 0.29 | 22% | Home go_to_screen fan-out |
| noise | -1 | *Unassigned* | 1770 | 6 | 0.05 | 0.22 | 13% | no dense neighbourhood |

---

## Artefacts

| file | contents |
|---|---|
| `src/Rule_based/cluster_names.py` | the name table (edit here) |
| `outputs/clusters/{platform}_{route}_catalog_named.csv` | catalog + family / name / note |
| `outputs/clusters/{platform}_{route}_labels_named.csv` | every journey with its cluster name |
| `outputs/clusters/{platform}_{route}_families.csv` | family roll-up |

Applying names in code:

```python
from Rule_based.cluster_names import apply_names, family_summary, lookup

lookup("ios", "a_tfidf", 40)          # -> ('econtract', 'E-contract list thrash', '38 steps, ...')
apply_names(scored, "ios", "a_tfidf")  # adds family / cluster_name / naming_note
family_summary(catalog, "ios", "a_tfidf")
```
