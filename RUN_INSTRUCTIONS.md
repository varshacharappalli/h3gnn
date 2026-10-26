Run these commands in PowerShell from this folder. Training is started only when you execute the training command.

Architecture preservation: the model copy differs from the official model only in three expressions that use the actual batch size for partial batches. Tensor summation, CUDA allocations and normalization formulas otherwise match the official source. Graph construction calls upstream.util.data_masks directly, including the original dense item matrix and the original subsequent sparse conversion. No custom sparse graph construction remains. Evaluation, checkpoint selection, logging and batching changes are retained. The Diginetica dataset preparation remains as documented below. Run `python verify_architecture.py` for a static structural audit without training.

Compatibility status: removing the earlier compatibility edits also restores the official `np.sum(final, 0)` over gradient-bearing CUDA tensors. This is incompatible with the installed modern NumPy/PyTorch environment; the command below must not be treated as verified runnable until the original environment is made compatible or that specific fix is authorized. No training was run after restoring the original computation.

```powershell
Set-Location 'C:\Users\Varsha Charapalli\OneDrive\Documents\ChatGPT\h3gnn'
.\.venv\Scripts\python.exe -u run.py --data data/diginetica --output runs/diginetica-seed42 --device 0 --seed 42 --epochs 30 --patience 5 --lr 0.001 --batch-size 100 --embedding-size 300 --layers 1 --pair-layers 2
```

Data has already been downloaded and prepared. Use a different `--output` directory for each new experiment. Existing epoch results are protected from accidental overwriting. This command starts a fresh training run; automatic resume is not implemented.

Outputs in `runs/diginetica-seed42/`:

- `training.log`: timestamps, progress every 100 batches, epoch results, best checkpoint updates, early stopping and final test results. Appended and flushed as each log event arrives.
- `epochs.csv`: training and validation losses and HR@10, HR@20, NDCG@10 and NDCG@20 after every completed epoch.
- `best.pt`: model parameters, optimizer state, best epoch, validation metrics, configuration and random generator states. Replaced atomically on each strict improvement in validation NDCG@20; ties retain the earlier epoch.
- `config.json`: configuration, dataset metadata, versions, source revision and model hash.
- `test_metrics.json` and `results.md`: final test metrics/table from the best validation checkpoint. Written after training finishes or early stops.

Watch the log from a second PowerShell window:

```powershell
Get-Content .\runs\diginetica-seed42\training.log -Wait -Tail 20
```

Re-evaluate the saved checkpoint without training (use the same model arguments as training):

```powershell
.\.venv\Scripts\python.exe -u run.py --data data/diginetica --output runs/diginetica-seed42 --device 0 --seed 42 --lr 0.001 --batch-size 100 --embedding-size 300 --layers 1 --pair-layers 2 --evaluate
```

All metrics are fractions from 0 to 1, with full-catalog ranking and one relevant next item per prefix. HR is 1 when the target is within the top K; NDCG is 1/log2(rank+1) when rank <= K and zero otherwise. Multiply by 100 for percentages. Testing is performed after selecting the best validation epoch; it does not select the checkpoint. The optimal epoch is unknown until this run completes. Thirty epochs with patience five is a starting policy, not an empirically established optimum.

Protocol caveat: this is an adaptation to user-identified Diginetica, not a claimed reproduction of paper numbers. Anonymous sessions are excluded; sessions with one known user inherit that identity for missing-user rows, and sessions with conflicting known users are excluded. Validation/test users must occur in training. Train-only item support threshold is five; unseen items are removed, and sessions shorter than two or longer than 999 retained clicks are excluded. All prefixes are generated after session-level chronological splitting. Training precedes 2016-05-19, validation spans 2016-05-19 through 2016-05-25, and testing spans 2016-05-26 through 2016-06-01. The resulting counts are 53,962/163/179 sessions and 187,982/436/509 prefix examples. These small user-overlap holdouts should be considered when interpreting results. Exact metadata and raw SHA-256 are in `data/diginetica/metadata.json`.

The untouched official repository is in `upstream/`, commit `919ed7e51f9a22604ec2fcefb1525d6fdcfd5bca`. `patch_model.py` generates `model_compat.py` with only variable batch size edits. The model modules and hybrid attention computation are retained, including the original pair-layer loop behavior. `h3_data.py` calls the official dense graph builder and retains its normalization expressions and uses the upstream session encoder inputs. Validation/test graphs are never added to training graphs. The original source configures only lastfm, xing and reddit; its Diginetica help text is insufficient. Its validation samples overlap test data, and its final batches duplicate examples; this runner replaces that training/evaluation workflow.

Original generic defaults: learning rate 0.001, batch size 100, embedding size 300, 30 epochs, hypergraph layers 1, pair layers 2. Xing overrides embedding size to 200 and layers to 5; LastFM overrides learning rate to 0.0005. No dropout is applied by the default forward path. The `l2=1e-5` argument is stored but is not applied as optimizer weight decay by the official model; this behavior is retained. No Diginetica-specific optimum or paper test scores have been verified or claimed. No full Diginetica epoch has completed in this workspace.

Validation status after restoration: the static architecture audit passed. The earlier GPU smoke-test success applied to the previous compatibility copy and does not establish that the restored original computation runs in the installed environment. No training was launched.
