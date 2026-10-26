"""Meaningful regression checks for ranking, batching, graph parity and GPU gradients."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from h3_data import Examples, graphs
from run import metrics, evaluate
from model_compat import H3GNN, forward


class Checks(unittest.TestCase):
    def test_metrics(self):
        scores = torch.arange(25, 0, -1).float()[None].repeat(4, 1)
        result = metrics(scores, torch.tensor([0, 9, 19, 24]))
        self.assertEqual(result['HR@10'], 2)
        self.assertEqual(result['HR@20'], 3)
        self.assertAlmostEqual(result['NDCG@10'], 1+1/np.log2(11), places=6)
        self.assertAlmostEqual(result['NDCG@20'], 1+1/np.log2(11)+1/np.log2(21), places=6)

    def test_batches_and_graph(self):
        data = Examples([dict(user=0, items=[1, 2, 1, 2]), dict(user=1, items=[2, 3])])
        np.testing.assert_array_equal(np.concatenate(data.generate_batch(3)), np.arange(4))
        g = graphs(data, 3, 2)
        h = np.zeros((4, 3))
        for j, seq in enumerate(data.raw):
            h[j, np.unique(seq)-1] = 1
        bh = h / h.sum(1)[:, None]
        dh = h.T / np.maximum(h.sum(0)[:, None], 1)
        np.testing.assert_allclose(g.adjacency.toarray(), dh @ bh)
        np.testing.assert_allclose(g.u_adj.toarray(), np.array([[1/3]*3+[0], [0]*3+[1]]) @ bh)
        self.assertAlmostEqual(g.global_A.toarray()[0, 1], 1/3)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_gpu_forward_backward_partial_batch_and_eval_restore(self):
        data = Examples([dict(user=0, items=[1, 2, 3, 1]), dict(user=1, items=[2, 3])])
        g = graphs(data, 3, 2)
        opt = SimpleNamespace(ishybrid=True, ishyper=True, isfixeduser=False, isnorm=True,
                              ishist=False, hybrid=0, encoder=0, pair_layer=2)
        model = H3GNN(opt, g.adjacency, g.adjacency, g.u_adj, g.global_A,
                      3, 2, .001, 1, 1e-5, 'diginetica', 8, 3).cuda()
        for idx in data.generate_batch(3):
            tar, scores, _ = forward(model, idx, data)
            loss = model.loss_function(scores, tar)
            loss.backward()
            self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
            model.optimizer.step()
            model.optimizer.zero_grad()
        before = model.embedding.weight.detach().clone()
        result = evaluate(model, data)
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
        self.assertTrue(torch.equal(before, model.embedding.weight))


if __name__ == '__main__':
    unittest.main()
