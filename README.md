# H3GNN on Diginetica

This workspace adapts **H3GNN (Hybrid Hierarchical HyperGraph Neural Network)** for personalized next-item recommendation on Diginetica. Given a real user and the items already clicked in a session, the model ranks possible next items.

The local completed experiment achieved **HR@10 = 49.12%**, **HR@20 = 60.51%**, **NDCG@10 = 0.319540**, and **NDCG@20 = 0.348572**. These are results for this repository's filtered, user-aware Diginetica protocol, not claimed reproductions of the paper's benchmark scores.

The raw file contains **1,235,380 click rows in 310,324 sessions**. Preprocessing retains **243,231 clicks in 54,304 sessions** and removes **992,149 click rows (80.31%)**. The sections below explain every removal stage and how those sessions become training and evaluation examples.

## Repository guide

| Path | Purpose |
|---|---|
| `prepare.py` | Reads raw clicks, resolves session users, splits chronologically, filters items/sessions/users, and writes JSON data and metadata. |
| `h3_data.py` | Expands sessions into prefix/target examples, batches each example exactly once per pass, and calls the official graph builder. |
| `model_compat.py` | Local model copy used by `run.py`, with partial-batch support and a PyTorch tensor-summation compatibility fix. |
| `run.py` | Seeds the experiment, builds training graphs, trains, validates, selects a checkpoint, and evaluates the test split. |
| `patch_model.py` | Generates a model copy with three partial-batch edits. **It does not apply the additional tensor-summation fix currently present in `model_compat.py`.** |
| `verify_architecture.py` | Checks the pinned upstream revision and preservation of model classes, constructors, and interfaces. This is a structural check, not a full computation-equivalence test. |
| `test_reproduction.py` | Checks ranking metrics, batch coverage, graph calculations, and an optional CUDA forward/backward/evaluation pass. |
| `requirements-lock.txt` | Snapshot of installed package versions, including unrelated environment packages; not a minimal dependency list. |
| `RUN_INSTRUCTIONS.md` | PowerShell commands and output-file guide; current status is clarified below. |
| `upstream/` | Local official source checkout at commit `919ed7e51f9a22604ec2fcefb1525d6fdcfd5bca`. |
| `data/raw/train-item-views.csv` | Raw semicolon-separated Diginetica click file. |
| `data/diginetica/` | Prepared `train.json`, `valid.json`, `test.json`, `metadata.json`, and `id_maps.json`. |
| `data/hypers2rec_exact/` | Separate preserved dataset copy without the user identities needed by this runner. |
| `runs/` | Experiment configurations, epoch logs, checkpoints, and final results. |

The official source is [Williamy946/H3GNN](https://github.com/Williamy946/H3GNN). Its README describes the personalized session recommendation model and the original Python/PyTorch 1.6.0/NumPy 1.18.1 environment. The local runner uses a different data protocol and modern environment.

`data/`, `runs/`, `upstream/`, `.venv/`, and `__pycache__/` are ignored by the root `.gitignore`. A fresh GitHub checkout therefore needs the upstream checkout and data supplied separately. No root Git remote is currently configured in this workspace; the changes described here are local files, not a published GitHub update.

## What a row, session, and example mean

- A **raw row** is one item-view/click event in the CSV, excluding the header.
- A **session** is a group of rows sharing `sessionId`.
- A **retained click** is an event that survives filtering in a saved session.
- A **prediction example** is a prefix of a retained session paired with its next item. A session with `L` retained clicks creates `L - 1` examples.

These counts are different. A saved JSON record represents a session, not one CSV row or one prediction example. Repeated clicks are kept; there is no duplicate-row removal or consecutive-repeat collapse.

## Preprocessing, in execution order

### 1. Read and order raw clicks

`prepare.py` reads UTF-8 CSV with optional byte-order mark and delimiter `;`. It uses `sessionId`, `userId`, `itemId`, `eventdate`, and integer `timeframe`. Rows are grouped by session, then ordered by `(eventdate, timeframe, original row index)`. The original index resolves exact ordering ties. A session's date is its maximum event date.

The raw file has **122,993 distinct item IDs**. Its SHA-256 is:

```text
0b9edf1b7960bb855a2248972f8ed593f18721f321b976b2077c9f782fb43e38
```

### 2. Resolve a real user for each session

An ID is missing if `userId.strip().upper()` is empty, `NA`, `NAN`, or `NULL`. There are **862,389 missing-user rows**, but this is **not** the number of rows removed for missing identity.

- No known user in a session: remove the entire session.
- More than one distinct known user: remove the entire session because its identity is ambiguous.
- Exactly one known user: assign that user to the whole session, including clicks whose user field is missing.
- After identity checks, remove sessions with fewer than two original clicks.

The code tests missingness using stripped/uppercased text, but keeps known user strings as written when comparing and mapping IDs. It does not manufacture users or substitute session IDs for users. Of the sessions eligible for chronological splitting, **1,862 missing-user clicks** receive their session's single known identity; some may be removed by later filters.

| Initial exclusion reason, applied sequentially | Sessions removed | Click rows removed |
|---|---:|---:|
| Entirely anonymous session | 216,218 | 860,521 |
| Conflicting known users | 194 | 1,293 |
| Fewer than two original clicks after the above checks | 27,838 | 27,838 |
| **Total initial exclusions** | **244,250** | **889,652** |
| **Remaining for splitting** | **66,074** | **345,728** |

### 3. Split whole sessions by date

The latest eligible session date is `2016-06-01`. The last seven calendar days form the test interval, and the preceding seven days form validation. Sessions are assigned before prefix expansion, so prefixes of a session cannot be split across train/validation/test.

| Split | Session date | Sessions before item filtering | Clicks before item filtering |
|---|---|---:|---:|
| Train | Before `2016-05-19` | 64,676 | 338,067 |
| Validation | `2016-05-19` through `2016-05-25` | 618 | 3,458 |
| Test | `2016-05-26` through `2016-06-01` | 780 | 4,203 |

The intervals are computed from the data rather than hard-coded. Cross-day sessions belong to the interval containing their last event day. Within the saved files, sessions are ordered by mapped user, day, and session ID; session IDs are strings, so same-day session ordering is lexical rather than a recovered start-time ordering.

### 4. Filter items using training clicks only

Count each item occurrence in the initial training split. Keep items with at least **five occurrences**, including repeat clicks. This is not a count of distinct users or sessions. Apply that same vocabulary to all three splits, removing rare and unseen items from each sequence.

The threshold is computed once. It is not repeatedly recalculated after short sessions are removed. Removing an intermediate click joins its surviving neighbors, so some resulting transitions span removed events.

### 5. Filter session lengths, then holdout users

After item filtering, keep sessions with **2 to 999 retained clicks**, inclusive. The upper bound is a preprocessing rule compatible with the model's 1,000-position embedding table. No session exceeds it in this dataset.

Build user and item maps from the remaining training sessions. Remove validation/test sessions whose users do not occur in those training sessions. This supports the upstream user/item normalization, which assumes observed training graph connections, and restricts evaluation to known users.

Finally, map users to `0..51,619` and items to `1..17,006`, with item ID `0` reserved for padding. Both maps use lexically sorted raw string IDs. The output-writing step filters items against the final training item map again and requires at least two clicks; **that final step removes no additional clicks or sessions in this dataset**.

### Detailed split audit

The following counts were recomputed from the raw CSV using the current preprocessing rules. Reconstructed records were compared against all three saved split files and matched exactly. A row removed by the item filter is counted only there; the later session-removal counts include only the clicks remaining at that stage.

| Stage | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Clicks entering split | 338,067 | 3,458 | 4,203 | 345,728 |
| Clicks removed by item support/vocabulary filter | 89,126 | 1,989 | 2,229 | 93,344 |
| Sessions removed for fewer than 2 retained clicks | 10,714 | 250 | 290 | 11,254 |
| Remaining clicks removed with those short sessions | 6,997 | 122 | 150 | 7,269 |
| Sessions removed for more than 999 clicks | 0 | 0 | 0 | 0 |
| Sessions after length filter | 53,962 | 368 | 490 | 54,820 |
| Sessions removed for users absent from training | 0 | 205 | 311 | 516 |
| Clicks removed with those unknown-user sessions | 0 | 748 | 1,136 | 1,884 |
| **Final sessions** | **53,962** | **163** | **179** | **54,304** |
| **Final retained clicks** | **241,944** | **599** | **688** | **243,231** |
| **Prefix/target examples** | **187,982** | **436** | **509** | **188,927** |

Some sessions become empty after item filtering; removing them adds zero to the subsequent dropped-click count. This explains why 11,254 removed short sessions account for only 7,269 additional clicks.

The complete row accounting is:

```text
1,235,380 raw clicks
  - 860,521 clicks in anonymous sessions
  -   1,293 clicks in conflicting-user sessions
  -  27,838 clicks in original singleton sessions
  -  93,344 clicks removed by item filtering
  -   7,269 surviving clicks in newly short sessions
  -   1,884 clicks in holdout sessions with unknown users
  = 243,231 retained clicks
```

Thus **992,149 rows were dropped (80.31%)**, and **19.69%** were retained. Overall **256,020 sessions were dropped (82.50%)**. The final catalog has **17,006 items** and **51,620 training users**. Metadata stores the final dimensions and selected exclusions, but does not store all intermediate row counts; the table above comes from the raw-data audit.

## Prefix expansion and batching

For a retained session `[A, B, C, D]`, `h3_data.Examples` creates, in this order:

```text
[A, B, C] -> D
[A, B]    -> C
[A]       -> B
```

Each example retains the session's real mapped user. Targets remain one-based in the dataset; upstream `get_slice` subtracts one for cross-entropy and score-column indexing. Batch preparation adds padding, masks, unique session nodes, reversed sequence indices, and local adjacency information.

Training shuffles example indices each epoch. Evaluation uses their saved order. The last batch may be smaller than the configured batch size; examples are neither duplicated nor discarded to fill it. Metrics are aggregated over individual examples, so longer sessions contribute more predictions than shorter sessions.

## How the model and graphs work

The default model combines two item-representation branches and a personalized session encoder:

1. **Hypergraph branch:** each training prefix is treated as a hyperedge containing its distinct items. Item co-membership propagates information between items; prefix-to-user membership aggregates information into user representations.
2. **Directed global item graph:** adjacent clicks in training prefixes create directed item connections, with self-connections initialized by the official builder. This branch captures item-transition information.
3. **Hybrid attention:** a learned sigmoid gate combines the hypergraph and global-graph item representations.
4. **Session encoding:** position-aware attention over the current sequence is combined with last-item information and the user representation.
5. **Scoring:** the session representation is multiplied by every catalog item's embedding to produce one score per candidate item.

`graphs()` calls `upstream.util.data_masks` directly on the **187,982 training prefixes**, not the original 53,962 full training sessions. Consequently, related prefixes contribute separate hyperedges, and common transitions can be counted repeatedly.

If `H` is the prefix-by-item binary incidence matrix and `U` is the prefix-by-user incidence matrix, the adapter forms:

```text
B = row_normalize(H)
D = row_normalize(H.T)
item adjacency = D @ B
user-item matrix = column_normalize(U).T @ B
```

The official global graph builder stores binary edges but increments its source normalization counter on every observed transition occurrence. Its resulting edge weights should not be described as conventional empirical transition probabilities or guaranteed row sums of one.

Only training prefixes build these shared graphs. Validation/test clicks are not added. The graphs summarize the entire training partition rather than a separate historical snapshot for each training example. Explicit historical-session sequence modeling is disabled (`ishist=False`); personalization still uses the training user-item aggregation.

Important implementation details:

- Hypergraph output sums the initial and propagated embedding layers; the current computation does not average them.
- The official pair-layer loop repeatedly multiplies the graph by the same initial embeddings. `pair_layers=2` therefore does not implement two recursively composed propagation hops. This behavior is preserved.
- The default forward path does not apply dropout. Although `l2=1e-5` is passed and stored, Adam has no configured weight decay and the runner adds no L2 penalty.
- The graph builder first allocates a dense `17,006 x 17,006` matrix before converting it to sparse format, roughly 2.31 GB for that float64 matrix alone, excluding other allocations.

## Hit ratio, NDCG, and loss

Each example has one relevant next item. `run.metrics()` ranks **all 17,006 catalog items**, with no sampled negatives and no masking of previously clicked items. Ties are resolved stably in favor of the lower score-column/item ID.

For target rank `r` (one-based):

```text
Hit@K(example)  = 1 if r <= K, otherwise 0
HR@K           = sum(Hit@K) / number_of_examples

NDCG@K(example) = 1 / log2(r + 1) if r <= K, otherwise 0
NDCG@K         = mean of those per-example values
```

With only one relevant target, ideal DCG is one, so no further denominator is needed. A rank-1 target contributes 1 to both metrics. A rank-10 target contributes 1 to HR@10 and about 0.2891 to NDCG@10. A rank-11 target contributes zero at K=10 but contributes a hit at K=20. NDCG rewards placing the hit earlier.

`metrics()` returns batch sums; `evaluate()` divides their accumulated sums by the total number of examples, including the partial batch correctly. Saved ranking metrics are **fractions**, not percentage values. Multiply by 100 when reporting HR as a percentage. Cross-entropy loss is a separate lower-is-better measure of next-item score quality; it is not a hit ratio.

Evaluation disables gradients and restores the model's item embedding weights afterward because the official forward path normalizes those weights in place even during evaluation.

## Actual completed experiment

Source artifacts: `runs/diginetica-run1/config.json`, `epochs.csv`, `training.log`, and `test_metrics.json`. The run completed on **2026-10-22**, with a maximum of 10 epochs and patience 5. It stopped after epoch 7; **epoch 2** had the highest validation NDCG@20, and its checkpoint produced the final test results.

| Metric | Validation at selected epoch | Test | Test interpretation |
|---|---:|---:|---|
| HR@10 | 0.467890 | 0.491159 | 250 / 509 targets hit; **49.12%** |
| HR@20 | 0.580275 | 0.605108 | 308 / 509 targets hit; **60.51%** |
| Cross-entropy loss | 5.021995 | 4.944966 | Lower is better |

At the selected epoch, training HR@10 was 0.697673, HR@20 was 0.823089, NDCG@10 was 0.390941, and NDCG@20 was 0.422853. These are evaluation passes over training examples after the epoch, not held-out generalization scores.

Validation NDCG@20 by epoch:

| Epoch | Validation NDCG@20 |
|---|---:|
| 1 | 0.312717 |
| **2** | **0.318580** |
| 3 | 0.311090 |
| 4 | 0.301970 |
| 5 | 0.286681 |
| 6 | 0.260508 |
| 7 | 0.226979 |

Training metrics continued improving while validation declined after epoch 2, which is consistent with overfitting. Checkpoint selection uses strict improvement in **validation NDCG@20**, retaining the first epoch on ties. Five subsequent non-improving epochs trigger stopping. Test metrics do not choose the checkpoint.

| Setting | Completed run value |
|---|---|
| Seed | 42 |
| Optimizer / learning rate | Adam / 0.001 |
| Batch size | 100 |
| Embedding dimension | 300 |
| Hypergraph layers / pair-layer setting | 1 / 2 |
| Maximum epochs / patience | 10 / 5 |
| PyTorch / NumPy | 2.11.0+cu128 / 2.5.2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |

The recorded model SHA-256 matches the current `model_compat.py`:

```text
896058792e5e3b1e6304cf85f936beb17710ccc90d51ea56c43e817d470abb6f
```

`runs/diginetica-seed42` is a separate interrupted run with one completed epoch and no final test result. `runs/diginetica-original-seed42` contains a startup log but no completed metrics. Do not combine these directories into one experiment.

## Reproduce preparation and training

Run from this repository's root in PowerShell. A fresh checkout needs Python, Git, a compatible CUDA PyTorch installation, NumPy, SciPy, and tqdm. The provided lock file records the existing environment; its CUDA wheel version may require the corresponding PyTorch wheel source. `run.py` explicitly requires CUDA.

If the ignored official checkout is absent:

```powershell
git clone https://github.com/Williamy946/H3GNN.git upstream
git -C upstream checkout 919ed7e51f9a22604ec2fcefb1525d6fdcfd5bca
```

Place the raw CSV at `data/raw/train-item-views.csv`. Its download provenance is not recorded in the preparation script; compare its SHA-256 with the value above before expecting identical counts.

```powershell
.\.venv\Scripts\python.exe prepare.py --raw data/raw/train-item-views.csv --output data/diginetica --min-support 5
.\.venv\Scripts\python.exe verify_architecture.py
.\.venv\Scripts\python.exe -m unittest test_reproduction.py
```

Preparation overwrites files in its chosen output folder. Keep the current `model_compat.py` when reproducing the completed run: invoking `patch_model.py` alone would remove its extra tensor-summation fix.

Start a new experiment with a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -u run.py --data data/diginetica --output runs/diginetica-reproduction-seed42 --device 0 --seed 42 --epochs 10 --patience 5 --lr 0.001 --batch-size 100 --embedding-size 300 --layers 1 --pair-layers 2
```

Omitting `--epochs` uses the runner's default of 30, rather than the completed experiment's maximum of 10. The runner seeds Python, NumPy, and CUDA/PyTorch, disables cuDNN benchmarking, and requires deterministic algorithms; unsupported nondeterministic operations cause failure. This does not establish reproducibility across different hardware/software environments.

Monitor progress in a second PowerShell window:

```powershell
Get-Content .\runs\diginetica-reproduction-seed42\training.log -Wait -Tail 20
```

Evaluate the existing completed checkpoint without retraining:

```powershell
.\.venv\Scripts\python.exe -u run.py --data data/diginetica --output runs/diginetica-run1 --device 0 --seed 42 --lr 0.001 --batch-size 100 --embedding-size 300 --layers 1 --pair-layers 2 --evaluate
```

Evaluation rewrites `test_metrics.json` and `results.md` in that output directory. Only load trusted local checkpoints: `torch.load(..., weights_only=False)` can execute pickle content. The runner checks selected model arguments and dataset metadata against the checkpoint, but does not enforce equality of all recorded software versions or source hashes.

## Experiment output files

| File | Meaning |
|---|---|
| `config.json` | Arguments, preprocessing metadata, library versions, GPU, upstream commit, model hash, and selection policy. |
| `training.log` | Timestamped startup, batch progress, epoch metrics, checkpoint selection, stopping, and errors. |
| `epochs.csv` | One row per completed epoch, containing update-time training loss and post-epoch training/validation metrics. |
| `best.pt` | Best validation model, optimizer, epoch, configuration, validation scores, and random-generator states; replacement is atomic. |
| `test_metrics.json` | Selected epoch, its validation metrics, and final test metrics. |
| `results.md` | Compact final test table. |

`train_update_loss` averages losses while parameters change during training. `train_loss` evaluates the resulting model after the epoch, so the two need not match. Existing epoch CSV files prevent starting a new training run in the same directory. Automatic checkpoint resume is not implemented even though optimizer/RNG states are saved.

## Compatibility and interpretation limits

The current model differs from the pinned upstream model in three expressions that use actual batch size and one expression replacing `np.sum(final, 0)` with `torch.stack(final, dim=0).sum(dim=0)` in `HyperConv`. The latter keeps summation in PyTorch with gradients and CUDA support. An additional `np.sum` remains in the unused `LineConv` path; its presence does not imply that the default path executes it. The current model constructors and parameter definitions match the upstream structural audit.

The completed-run artifacts supersede the older claim that no full epoch completed. Their model hash also supersedes the older description that only three partial-batch expressions differ. This documentation update does not change model behavior or run another training experiment.

The test set is only **179 sessions / 509 prefixes**, and validation is **163 sessions / 436 prefixes**. Prefixes within a session are related, so they are not 509 independent sessions. Excluding anonymous sessions, rare items, short sessions, and unseen users substantially changes the evaluated population. These scores describe known-user recommendation on this filtered dataset, not all original Diginetica traffic or cold-start performance. A single seed and small holdouts do not establish statistical significance or state-of-the-art performance.

The separate `data/hypers2rec_exact` manifest records 719,470 already-expanded training examples and 60,858 test examples. Those files lack user/session IDs and are not inputs to `run.py`; they must not be expanded again or treated as equivalent to the user-aware protocol above. Their counts and any results from a different protocol are not directly comparable to this experiment.
