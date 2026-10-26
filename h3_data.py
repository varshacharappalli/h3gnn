"""Official graph construction, with a dataset adapter and exact-once batching."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from upstream.util import Data as OriginalData, data_masks


class Examples(OriginalData):
    def __init__(self, records):
        seqs, targets, users = [], [], []
        for r in records:
            for j in range(len(r['items'])-1, 0, -1):
                seqs.append(r['items'][:j])
                targets.append(r['items'][j])
                users.append(r['user'])
        self.raw = np.empty(len(seqs), dtype=object)
        self.raw[:] = seqs
        self.targets = np.asarray(targets)
        self.user = np.asarray(users)
        self.length = len(seqs)
        # ishist=False in the official default; these values are unused by HyperConv.
        self.hist_sess = np.zeros((self.length, 1), dtype=np.int64)
        self.hist_len = np.zeros(self.length, dtype=np.int64)

    def generate_batch(self, batch_size, shuffle=False):
        order = np.random.permutation(self.length) if shuffle else np.arange(self.length)
        return [order[i:i+batch_size] for i in range(0, self.length, batch_size)]


def graphs(train, n_node, n_user):
    # Call the official function directly, including its dense np.diag allocation
    # and its own subsequent CSR conversion. No custom sparse replacement.
    h_t, u_h_t, global_a, hist, hist_len, _ = data_masks(
        train.raw, train.user, n_node, n_user, 1000, None, 0)
    train.hist_sess, train.hist_len = hist, hist_len
    # These are the original Data.__init__ normalization expressions.
    bh_t = h_t.T.multiply(1.0 / h_t.sum(axis=1).reshape(1, -1)).T
    h = h_t.T
    dh = h.T.multiply(1.0 / h.sum(axis=1).reshape(1, -1)).T
    adjacency = np.dot(dh, bh_t)
    u_h = u_h_t.T
    u_uh_t = u_h_t.multiply(1.0 / u_h.sum(axis=1).reshape(1, -1))
    u_item_matrix = u_uh_t.T @ bh_t
    return SimpleNamespace(adjacency=adjacency.tocoo(), u_adj=u_item_matrix.tocoo(),
                           global_A=global_a.tocoo())


def load(root, split):
    return Examples(json.loads((Path(root)/(split+'.json')).read_text(encoding='utf-8')))
