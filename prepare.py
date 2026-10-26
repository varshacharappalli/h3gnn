"""User-aware chronological Diginetica protocol; no synthetic user identities."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


def prepare(raw, output, min_support=5):
    sessions = defaultdict(list)
    missing = 0
    with open(raw, newline='', encoding='utf-8-sig') as f:
        for index, row in enumerate(csv.DictReader(f, delimiter=';')):
            if row['userId'].strip().upper() in ('', 'NA', 'NAN', 'NULL'):
                missing += 1
            sessions[row['sessionId']].append((row['eventdate'], int(row['timeframe']), index,
                                               row['itemId'], row['userId']))
    records = []
    anonymous_sessions = ambiguous_sessions = 0
    for sid, rows in sessions.items():
        rows.sort(key=lambda r: r[:3])
        known_users = {r[4] for r in rows if r[4].strip().upper() not in ('', 'NA', 'NAN', 'NULL')}
        if not known_users:
            anonymous_sessions += 1
            continue
        if len(known_users) > 1:
            ambiguous_sessions += 1
            continue
        user = next(iter(known_users))
        if len(rows) >= 2:
            records.append(dict(sid=sid, user=user, day=max(r[0] for r in rows),
                                items=[r[3] for r in rows]))
    last = max(date.fromisoformat(r['day']) for r in records)
    test_start = last - timedelta(days=6)
    valid_start = test_start - timedelta(days=7)
    buckets = [[], [], []]
    for r in records:
        d = date.fromisoformat(r['day'])
        buckets[0 if d < valid_start else 1 if d < test_start else 2].append(r)
    support = Counter(i for r in buckets[0] for i in r['items'])
    kept = {i for i, n in support.items() if n >= min_support}
    buckets = [[dict(r, items=[i for i in r['items'] if i in kept]) for r in b] for b in buckets]
    buckets = [[r for r in b if 2 <= len(r['items']) <= 999] for b in buckets]
    # Require train-observed users and items: zero-degree normalization is undefined upstream.
    users = sorted({r['user'] for r in buckets[0]})
    items = sorted({i for r in buckets[0] for i in r['items']})
    umap, imap = {u: n for n, u in enumerate(users)}, {i: n+1 for n, i in enumerate(items)}
    excluded = []
    for k in (1, 2):
        before = len(buckets[k])
        buckets[k] = [r for r in buckets[k] if r['user'] in umap]
        excluded.append(before-len(buckets[k]))
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, bucket in zip(('train', 'valid', 'test'), buckets):
        result = []
        for r in sorted(bucket, key=lambda r: (umap[r['user']], r['day'], r['sid'])):
            seq = [imap[i] for i in r['items'] if i in imap]
            if len(seq) >= 2:
                result.append(dict(user=umap[r['user']], items=seq, session=r['sid'], day=r['day']))
        if not result:
            raise ValueError('Empty ' + name + ' split')
        (out / (name+'.json')).write_text(json.dumps(result), encoding='utf-8')
        counts[name] = dict(sessions=len(result), examples=sum(len(r['items'])-1 for r in result))
    meta = dict(protocol='user-aware-chronological-v1', raw_sha256=hashlib.sha256(Path(raw).read_bytes()).hexdigest(),
                n_node=len(items), n_user=len(users), missing_user_rows=missing,
                anonymous_sessions_dropped=anonymous_sessions, ambiguous_sessions_dropped=ambiguous_sessions,
                valid_start=str(valid_start), test_start=str(test_start), last_day=str(last),
                min_train_item_support=min_support, unknown_user_sessions_dropped=excluded,
                max_session_length=999, counts=counts)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    (out/'id_maps.json').write_text(json.dumps(dict(users=umap, items=imap)), encoding='utf-8')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default='data/raw/train-item-views.csv')
    p.add_argument('--output', default='data/diginetica')
    p.add_argument('--min-support', type=int, default=5)
    a = p.parse_args()
    prepare(a.raw, a.output, a.min_support)
