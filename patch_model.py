"""Generate a reviewable compatibility copy; upstream checkout remains untouched."""
import hashlib
from pathlib import Path


def patch():
    source = Path('upstream/model.py').read_text(encoding='utf-8')
    changes = {
        'torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)':
            'torch.cuda.FloatTensor(session_item.shape[0], list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)',
        'range(self.batch_size)': 'range(session_item.shape[0])',
        'repeat(self.batch_size, 1, 1)': 'repeat(session_item.shape[0], 1, 1)',
    }
    for old, new in changes.items():
        assert old in source, old
        source = source.replace(old, new)
    Path('model_compat.py').write_text(source, encoding='utf-8')
    print('model_compat.py sha256:', hashlib.sha256(source.encode()).hexdigest())


if __name__ == '__main__':
    patch()
