import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from h3_data import load, graphs
from model_compat import H3GNN, forward

log = logging.getLogger('h3gnn')


def metrics(scores, targets):
    # Stable tie order: lower contiguous item ID first; all catalog items ranked.
    rank = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :20]
    match = rank.eq(targets[:, None])
    out = {}
    for k in (10, 20):
        hit = match[:, :k].float()
        out['HR@'+str(k)] = hit.sum().item()
        discount = torch.log2(torch.arange(2, hit.shape[1]+2, device=hit.device).float())
        out['NDCG@'+str(k)] = (hit / discount).sum().item()
    return out


def evaluate(model, data):
    # Upstream forward normalizes embedding.weight in-place, even in eval mode.
    saved = model.embedding.weight.detach().clone()
    model.eval()
    total, loss, count = {}, 0., 0
    try:
        with torch.no_grad():
            for indices in data.generate_batch(model.batch_size):
                target, scores, _ = forward(model, indices, data)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite evaluation scores')
                loss += model.loss_function(scores, target).item()*len(indices)
                for key, value in metrics(scores, target).items():
                    total[key] = total.get(key, 0.) + value
                count += len(indices)
    finally:
        with torch.no_grad():
            model.embedding.weight.copy_(saved)
    return dict(loss=loss/count, **{k: v/count for k, v in total.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/diginetica')
    p.add_argument('--output', default='runs/diginetica-seed42')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--embedding-size', type=int, default=300)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--pair-layers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required. Install a compatible CUDA PyTorch wheel.')
    torch.cuda.set_device(a.device)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Fail rather than silently claim deterministic execution if an op lacks support.
    torch.use_deterministic_algorithms(True)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if not a.evaluate and (out/'epochs.csv').exists():
        raise FileExistsError('Use a fresh output directory to avoid mixing experiments')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(out/'training.log', mode='a', encoding='utf-8'),
                                  logging.StreamHandler()], force=True)
    log.info('Starting %s with arguments: %s', 'evaluation' if a.evaluate else 'training', vars(a))
    meta = json.loads((Path(a.data)/'metadata.json').read_text())
    train = load(a.data, 'train')
    graph = graphs(train, meta['n_node'], meta['n_user'])
    opt = SimpleNamespace(ishybrid=True, ishyper=True, isfixeduser=False, isnorm=True,
                          ishist=False, hybrid=0, encoder=0, pair_layer=a.pair_layers, dataset='diginetica')
    model = H3GNN(opt, graph.adjacency, graph.adjacency, graph.u_adj, graph.global_A,
                  meta['n_node'], meta['n_user'], a.lr, a.layers, 1e-5, 'diginetica',
                  a.embedding_size, a.batch_size).cuda()
    config = dict(args=vars(a), metadata=meta, torch=torch.__version__, numpy=np.__version__,
                  gpu=torch.cuda.get_device_name(), upstream_commit=subprocess.check_output(
                      ['git', '-C', 'upstream', 'rev-parse', 'HEAD'], text=True).strip(),
                  model_sha256=hashlib.sha256(Path('model_compat.py').read_bytes()).hexdigest(),
                  selection='validation NDCG@20, strict improvement, first epoch wins ties',
                  metric_scale='fractions in [0,1]')
    if not a.evaluate:
        (out/'config.json').write_text(json.dumps(config, indent=2))
        valid = load(a.data, 'valid')
        best, stale, best_epoch = -float('inf'), 0, None
        for epoch in range(1, a.epochs+1):
            log.info('Epoch %d/%d started', epoch, a.epochs)
            start = time.monotonic()
            model.train()
            train_loss = 0.
            batches = train.generate_batch(a.batch_size, shuffle=True)
            for step, idx in enumerate(batches, 1):
                model.optimizer.zero_grad(set_to_none=True)
                target, scores, _ = forward(model, idx, train)
                loss = model.loss_function(scores+1e-8, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                model.optimizer.step()
                train_loss += loss.item()*len(idx)
                if step == 1 or step % 100 == 0 or step == len(batches):
                    log.info('Epoch %d batch %d/%d loss=%.6f', epoch, step, len(batches), loss.item())
            log.info('Epoch %d computing training and validation metrics', epoch)
            tr, va = evaluate(model, train), evaluate(model, valid)
            row = dict(epoch=epoch, train_update_loss=train_loss/train.length,
                       **{'train_'+k: v for k, v in tr.items()},
                       **{'valid_'+k: v for k, v in va.items()}, seconds=time.monotonic()-start)
            with (out/'epochs.csv').open('a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
            log.info('Epoch results: %s', json.dumps(row))
            if va['NDCG@20'] > best:
                best, stale, best_epoch = va['NDCG@20'], 0, epoch
                torch.save(dict(model=model.state_dict(), optimizer=model.optimizer.state_dict(),
                                epoch=epoch, validation=va, config=config,
                                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                                numpy_rng=np.random.get_state(), python_rng=random.getstate()), out/'best.tmp.pt')
                (out/'best.tmp.pt').replace(out/'best.pt')
                log.info('Saved best.pt: epoch=%d validation NDCG@20=%.6f', epoch, best)
            else:
                stale += 1
            if stale >= a.patience:
                log.info('Early stopping after %d epochs without improvement', stale)
                break
        log.info('Best epoch: %d; validation NDCG@20=%.6f', best_epoch, best)
    checkpoint = torch.load(out/'best.pt', map_location='cuda', weights_only=False)
    # Checkpoint is a locally generated trusted artifact; never load untrusted pickle files.
    for key in ('lr', 'batch_size', 'embedding_size', 'layers', 'pair_layers', 'seed'):
        if checkpoint['config']['args'][key] != vars(a)[key]:
            raise ValueError('Checkpoint configuration mismatch: ' + key)
    if checkpoint['config']['metadata'] != meta:
        raise ValueError('Dataset metadata does not match checkpoint')
    model.load_state_dict(checkpoint['model'])
    result = dict(best_epoch=checkpoint['epoch'], validation=checkpoint['validation'],
                  test=evaluate(model, load(a.data, 'test')))
    (out/'test_metrics.json').write_text(json.dumps(result, indent=2))
    table = '| Metric | Test (fraction) |\n|---|---:|\n' + ''.join(
        f'| {k} | {result["test"][k]:.6f} |\n' for k in ('HR@10', 'HR@20', 'NDCG@10', 'NDCG@20'))
    (out/'results.md').write_text(f'Best epoch: {result["best_epoch"]}\n\n'+table)
    log.info('Final results: %s', json.dumps(result))
    log.info('\n%s', table)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.warning('Interrupted by user; any completed best checkpoint is preserved')
        raise
    except Exception:
        log.exception('Run failed')
        raise
