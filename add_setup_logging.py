from pathlib import Path
import ast
root = Path.home() / 'OneDrive' / 'Desktop' / 'h3gnn' / 'H3GNN'
p = root / 'util.py'
s = p.read_text(encoding='utf-8')
if 'def log_stage(' not in s:
    s = s.replace('import numpy as np', 'import time\nfrom datetime import datetime\n\ndef log_stage(message):\n    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)\n\nimport numpy as np', 1)
    changes = {
        '    global_A = np.diag(np.ones(n_node))': '    log_stage(f"Allocating dense item graph: {n_node:,} x {n_node:,} (~{n_node*n_node*8/1024**3:.2f} GiB)")\n    global_A = np.diag(np.ones(n_node))\n    log_stage("Dense item graph allocated")',
        '    hist_sess = np.array([[0]*max_hist_len for i in range(len(user))])': '    log_stage(f"Allocating session history: {len(user):,} examples x {max_hist_len:,} history slots")\n    hist_sess = np.array([[0]*max_hist_len for i in range(len(user))])\n    log_stage("Session history allocated")',
        '    for j in range(len(all_sessions)):': '    graph_started = time.perf_counter()\n    log_stage(f"Building session/user connections and item transitions for {len(all_sessions):,} examples")\n    for j in range(len(all_sessions)):',
        '    print("Out")': '        if (j + 1) % 10000 == 0 or j + 1 == len(all_sessions):\n            elapsed = time.perf_counter() - graph_started\n            rate = (j + 1) / max(elapsed, 1e-9)\n            remaining = (len(all_sessions) - j - 1) / max(rate, 1e-9)\n            log_stage(f"Graph examples: {j+1:,}/{len(all_sessions):,} ({100*(j+1)/len(all_sessions):.1f}%), elapsed {elapsed:.0f}s, ETA ~{remaining:.0f}s")\n    log_stage("Finding nonzero item transitions in dense graph (this scan may take time)")',
        '    print("non zero assign")': '    log_stage(f"Found {len(g_indices):,} transitions; converting dense graph to sparse CSR")',
        '    print("sparse")': '    log_stage("Sparse item graph built; computing transition weights")',
        '    print("val generating")': '    log_stage("Transition weights ready; assembling normalized item graph")',
        '    print("sparsing...")': '    log_stage("Building sparse session-item and session-user incidence matrices")',
        '    print(len(all_sessions))': '    log_stage(f"Incidence rows: {len(all_sessions):,}")',
        '    return matrix, u_matrix, global_A, hist_sess, hist_len, train_hist': '    log_stage(f"Incidence matrices ready: session-item {matrix.shape}, nnz={matrix.nnz:,}; session-user {u_matrix.shape}, nnz={u_matrix.nnz:,}")\n    return matrix, u_matrix, global_A, hist_sess, hist_len, train_hist',
        '        data = common_seq(data)': '        preparation_started = time.perf_counter()\n        split_name = "training" if train_hist is None else "evaluation"\n        log_stage(f"Preparing {split_name} data: expanding {len(data[0]):,} source sequences into prefixes")\n        data = common_seq(data)\n        log_stage(f"Prefix expansion complete: {len(data[0]):,} prediction examples")',
        '        print("Masking...")': '        log_stage(f"Masking: building {split_name} graphs and history")',
        '        print("Done")': '        log_stage("Graph construction complete; normalizing session-item incidence")',
        '        DHBH_T = np.dot(DH,BH_T)': '        log_stage("Computing item-item hypergraph propagation matrix")\n        DHBH_T = np.dot(DH,BH_T)\n        log_stage("Hypergraph propagation matrix ready; normalizing user incidence")',
        '        u_item_matrix = u_UH@BH_T': '        log_stage("Computing user-item aggregation matrix")\n        u_item_matrix = u_UH@BH_T',
        '        self.edge_item = BH_T.tocsr()': '        log_stage("User-item matrix ready; finalizing sparse graph formats")\n        self.edge_item = BH_T.tocsr()',
        '        self.shuffle = shuffle': '        self.shuffle = shuffle\n        log_stage(f"{split_name.capitalize()} data ready: {self.length:,} examples, total preparation {time.perf_counter()-preparation_started:.1f}s")',
    }
    for old, new in changes.items():
        if old not in s:
            raise RuntimeError(f'Missing expected source: {old}')
        s = s.replace(old, new, 1)
    ast.parse(s)
    p.write_text(s, encoding='utf-8')
ast.parse(p.read_text(encoding='utf-8'))
print('Progress logging added; syntax verified. No training started.')
