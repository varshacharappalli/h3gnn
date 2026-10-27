from pathlib import Path
import ast
root = Path.home() / 'OneDrive/Desktop/h3gnn/H3GNN'
helper = '''import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

RUN_DIR = None

class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def isatty(self):
        return False
    def __getattr__(self, name):
        return getattr(self.stream, name)

def setup_logs(options):
    global RUN_DIR
    RUN_DIR = Path('runs') / (options.dataset + '-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    RUN_DIR.mkdir(parents=True)
    file = (RUN_DIR / 'training.log').open('a', encoding='utf-8', buffering=1)
    sys.stdout = Tee(sys.stdout, file)
    sys.stderr = Tee(sys.stderr, file)
    (RUN_DIR / 'config.json').write_text(json.dumps(vars(options), indent=2), encoding='utf-8')
    print(f'[{datetime.now():%H:%M:%S}] Logs: {RUN_DIR.resolve()}', flush=True)

def record_metrics(filename, row):
    if RUN_DIR is None:
        return
    path = RUN_DIR / filename
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
'''
main_path, model_path = root / 'main.py', root / 'model.py'
a, b = main_path.read_text(encoding='utf-8'), model_path.read_text(encoding='utf-8')
if 'from run_logging import setup_logs' not in a:
    a = 'from run_logging import setup_logs, record_metrics\n' + a
    a = a.replace('    train_data = pickle.load', '    setup_logs(opt)\n    train_data = pickle.load', 1)
    a = a.replace("        print('epoch: ', epoch)", "        epoch_started = time.perf_counter()\n        print(f'Epoch {epoch+1}/{opt.epoch} started', flush=True)")
    needle = "        if best_results['metric10'][0] < metrics['hit10']:"
    addition = '''        elapsed = time.perf_counter() - epoch_started
        epoch_times.append(elapsed)
        remaining = sum(epoch_times) / len(epoch_times) * (opt.epoch - epoch - 1)
        record_metrics('epochs.csv', dict(epoch=epoch+1, seconds=elapsed, train_loss=total_loss, **metrics))
        print(f'Epoch {epoch+1}/{opt.epoch} complete | mean train loss {total_loss:.6f} | duration {elapsed/60:.1f} min | remaining ~{remaining/3600:.2f} hours', flush=True)
        print('Test (best validation checkpoint): ' + ' | '.join(f'{key}={value:.4f}%' for key, value in metrics.items()), flush=True)
'''
    a = a.replace(needle, addition + needle, 1)
    a = a.replace('    for epoch in range(opt.epoch):', '    epoch_times = []\n    for epoch in range(opt.epoch):', 1)
if 'from run_logging import record_metrics' not in b:
    b = 'from run_logging import record_metrics\nimport time\n' + b
    b = b.replace('    for i in tqdm(slices):', '    batch_started = time.perf_counter()\n    model.train()\n    for i in tqdm(slices, desc=f"Epoch {epoch+1} training", mininterval=10):', 1)
    b = b.replace('        par_5 = int(len(slices)/5)-1', '''        total_loss += loss.item()
        if ind == 1 or ind % 100 == 0 or ind == len(slices):
            elapsed = time.perf_counter() - batch_started
            eta = elapsed / ind * (len(slices) - ind)
            print(f'Epoch {epoch+1} | batch {ind}/{len(slices)} | loss {loss.item():.6f} | mean loss {total_loss/ind:.6f} | elapsed {elapsed/60:.1f} min | training ETA ~{eta/60:.1f} min (includes validation time so far)', flush=True)
        par_5 = max(1, int(len(slices)/5)-1)''', 1)
    b = b.replace('val_loss = model.loss_function(scores + 1e-8, targets)', 'val_loss = model.loss_function(scores + 1e-8, tar)', 1)
    marker = "            if (metrics['hit10']+metrics['mrr10'] > best_perform):"
    b = b.replace(marker, "            record_metrics('validation.csv', dict(epoch=epoch+1, batch=ind, train_loss_mean=total_loss/ind, **metrics))\n" + marker, 1)
    b = b.replace('        #total_loss += loss', '        model.train()\n        #total_loss += loss', 1)
    b = b.replace('best_model = torch.load("./ckpt/"+model.opt.dataset+"_ckpt.pt")', 'best_model = torch.load("./ckpt/"+model.opt.dataset+"_ckpt.pt", weights_only=False)', 1)
    b = b.replace('    return metrics, total_loss, best_perform', '    return metrics, total_loss / max(updates_per_epoch, 1), best_perform', 1)
for text in [helper, a, b]:
    ast.parse(text)
(root / 'run_logging.py').write_text(helper, encoding='utf-8')
main_path.write_text(a, encoding='utf-8')
model_path.write_text(b, encoding='utf-8')
print('Persistent logging and metric CSVs added. Syntax checks passed; no training started.')
