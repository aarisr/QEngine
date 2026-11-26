#!/usr/bin/env python3
"""slotlog_to_qtrace.py: QEngine flash slot log (kubectl logs --timestamps, FLASH_SLOT_LOG=1) -> a qtrace v0 directory.

usage: slotlog_to_qtrace.py LOG OUTDIR --instance A [--served-name NAME]
This is the "v-1" converter of the qtrace spec (section 6): the log carries no token ids, so the prompt structure
(q_parent, q_div, q_pin_class, q_hint) is synthesized from lengths and reuse counts, see parent_of() below.
"""
import argparse, json, os, re, statistics
from datetime import datetime

TS = re.compile(r'^(\S+)\s(.*)$')
START = re.compile(r'\[qflash\] start: (\d+) tokens \(reused (\d+), (\w+)\) in ([\d.]+) ms \(pool ([\d.]+), restore ([\d.]+)\), slot (\d+) \[\d+, \d+\), conversation (\d+), (\d+) known, ([\d.]+) GB host')
CHAT = re.compile(r'\[chat\] slot (\d+) prompt (\d+) \(cached (\d+)(?: from ([a-z ]+))?, prefill (\d+) tok/s\) \| gen (\d+) @ ([\d.]+) tok/s \| (\w+)(.*)')
THINK = re.compile(r'think (\d+)/(\d+) \(effort:(\w+)\)')
PIN = re.compile(r'\[qflash\] archive: pinned system prefix (\d+) tokens of slot (\d+)')
PINUSE = re.compile(r'\[qflash\] archive: slot (\d+) starts from the pinned system prefix \((\d+) tokens')
RELEASED = re.compile(r'\[qflash\] slot (\d+) released: conversation (\d+), (\d+) tokens, idle ([\d.]+) s -> (\w+)')
LOST = re.compile(r'\[qflash\] archive: lost conversation (\d+) \((\d+) tokens, idle ([\d.]+) s, prio (\d+), ([\d.]+) MB freed')
DROPPED = re.compile(r'\[qflash\] archive: (checkpoints|archived rows) dropped.*conversation (\d+) \((\d+) tokens, idle ([\d.]+) s, prio (\d+), ([\d.]+) MB freed')
SYNCED = re.compile(r'\[qflash\] archive: conversation (\d+) \(slot (\d+)\) synced at (\d+) tokens')
SRV = re.compile(r'==== qengine server start: QEngine (\S+), pid \d+, build (\w+)')
ENV = re.compile(r'(?:^|\s)env: (.*)$')
CKEVERY = re.compile(r'checkpoints every (\d+) tokens')
KNOBS = ['FLASH_MAX_SEQS', 'FLASH_KV_ARCHIVE_GB', 'FLASH_PREFIX_GB', 'FLASH_CKPT', 'FLASH_CKPT_MIN', 'FLASH_END_CUT',
         'FLASH_CKPT_KEEP', 'FLASH_KV', 'FLASH_KV_BITS', 'FLASH_DISK_CACHE_GB']
SRC = {'checkpoint': 'checkpoint', 'archive': 'archive', 'none': 'none', 'live': 'live'}


def ts(s):
    return datetime.fromisoformat(re.sub(r'(\.\d{6})\d*', r'\1', s)).timestamp()


def num(v):
    try:
        return float(v) if '.' in v else int(v)
    except ValueError:
        return v


def parse(path):
    meta, starts, evs = {'knobs': {}}, [], []   # starts: request dicts; evs: (t, kind, groups)
    pending, pin_for, use_for, bad = {}, {}, {}, 0
    for line in open(path, errors='replace'):
        m = TS.match(line.rstrip('\n'))
        if not m:
            continue
        t, msg = ts(m.group(1)), m.group(2)
        if (x := SRV.search(msg)):
            meta.setdefault('version', x.group(1)); meta.setdefault('build', x.group(2)); meta.setdefault('pod_start', t)
        elif (x := ENV.search(msg)) and 'FLASH_' in msg and 'env' not in meta:
            meta['env'] = dict(kv.split('=', 1) for kv in x.group(1).split() if '=' in kv)
            meta['cards'] = len(meta['env'].get('QE_DEVICES', '').split(',')) if 'QE_DEVICES' in meta['env'] else None
        elif (x := CKEVERY.search(msg)):
            meta.setdefault('ckpt_every', int(x.group(1)))
        elif (x := START.search(msg)):
            P, R, src, ms, pool, rest, slot, conv, known, host = x.groups()
            r = dict(t=t, P=int(P), reused=int(R), source=SRC.get(src, src), slot=int(slot), conv=int(conv), host_gb=float(host),
                     start_ms=float(ms), pinned=use_for.pop(int(slot), 0), pin_made=pin_for.pop(int(slot), 0), idx=len(starts))
            starts.append(r)
            pending[r['slot']] = r
        elif (x := PIN.search(msg)):
            pin_for[int(x.group(2))] = int(x.group(1)); evs.append((t, 'pin', x.groups()))
        elif (x := PINUSE.search(msg)):
            use_for[int(x.group(1))] = int(x.group(2)); evs.append((t, 'pin_use', x.groups()))
        elif (x := CHAT.search(msg)):
            slot, P, C, frm, pf, gen, gtps, stop, rest = x.groups()
            r = pending.pop(int(slot), None)
            if r is None:
                bad += 1
                continue
            th = THINK.search(rest)
            r.update(t_end=t, prefill_tps=int(pf), gen=int(gen), gen_tps=float(gtps), think=int(th.group(1)) if th else 0,
                     effort=th.group(3) if th else 'off', tool_required='tool_choice required' in rest)
        else:
            for kind, rx in (('released', RELEASED), ('lost', LOST), ('dropped', DROPPED), ('synced', SYNCED)):
                if (x := rx.search(msg)):
                    evs.append((t, kind, x.groups())); break
    meta['unmatched_chat'] = bad
    return meta, starts, evs


def parent_of(reqs):
    """Synthesize the prompt tree (no tokens in the log). Most recent earlier request i on the instance with
    P_i <= P_j, P_i - 1 >= reused_j, P_j - P_i < 16000 (a busy parent only offers older checkpoints, so the true parent can
    be the newer request, not the exact P_i-1 == reused match); only when reused_j exceeds the pinned class j itself shows."""
    for j, r in enumerate(reqs):
        r['parent'] = -1
        if r['reused'] <= r['cls0']:
            continue
        for i in range(j - 1, -1, -1):
            p = reqs[i]
            if p['P'] <= r['P'] and p['P'] - 1 >= r['reused'] and r['P'] - p['P'] < 16000:
                r['parent'] = i; break
    for r in reqs:   # parents precede children: classes propagate in one pass
        r['cls'] = r['cls0'] if r['cls0'] else (reqs[r['parent']]['cls'] if r['parent'] >= 0 else 0)
        r['div'] = reqs[r['parent']]['P'] - 1 if r['parent'] >= 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log'); ap.add_argument('out'); ap.add_argument('--instance', required=True)
    ap.add_argument('--served-name', default='qwen38-flash-next')
    a = ap.parse_args()
    meta, starts, evs = parse(a.log)
    reqs = [r for r in starts if 't_end' in r]
    dropped = len(starts) - len(reqs)
    if not reqs:
        raise SystemExit('no requests in log')
    t0 = reqs[0]['t']
    for n, r in enumerate(reqs):
        r['id'] = n
        r['cls0'] = r['pin_made'] or r['pinned']   # the pin line precedes the start line of the request it came from
    parent_of(reqs)
    ms = lambda t: round((t - t0) * 1000, 3)
    byslot = {}
    ids = {r['idx']: r['id'] for r in reqs}

    def next_start(t, slot):
        c = [r for r in starts if r['t'] >= t and r['idx'] in ids]
        s = [r for r in c if r['slot'] == slot]
        return ids[(s or c)[0]['idx']] if c else None

    dec = []
    for r in reqs:
        dec.append(dict(t=ms(r['t']), seam='S7', req=r['id'], kind=r['source'], L=r['reused'], conv=r['conv'], slot=r['slot']))
    slot_req = {}   # slot -> request id of the latest pin/pin_use line owner (the next start on that slot)
    for t, kind, g in evs:
        if kind == 'released':
            dec.append(dict(t=ms(t), seam='S9', kind='victim', req=next_start(t, int(g[0])), victim_slot=int(g[0]), victim_conv=int(g[1]),
                            tokens=int(g[2]), idle_s=float(g[3])))
        elif kind == 'lost':
            dec.append(dict(t=ms(t), seam='S2', kind='lost', req=None, conv=int(g[0]), tokens=int(g[1])))
        elif kind == 'dropped':
            dec.append(dict(t=ms(t), seam='S2', kind='drop_spare_states' if g[0] == 'checkpoints' else 'drop_rows', req=None,
                            conv=int(g[1]), tokens=int(g[2])))
        elif kind == 'synced':
            dec.append(dict(t=ms(t), seam='S5', kind='sync', req=None, conv=int(g[0]), slot=int(g[1]), tokens=int(g[2])))
        elif kind == 'pin':
            dec.append(dict(t=ms(t), seam='pin', kind='pin', req=next_start(t, int(g[1])), slot=int(g[1]), pos=int(g[0])))
        elif kind == 'pin_use':
            dec.append(dict(t=ms(t), seam='pin', kind='pin_use', req=next_start(t, int(g[0])), slot=int(g[0]), pos=int(g[1])))
    dec.sort(key=lambda d: d['t'])
    tps = [r['prefill_tps'] for r in reqs if r['prefill_tps'] > 0]
    env = meta.get('env', {})
    knobs = {k: num(env[k]) for k in KNOBS if k in env}
    if 'FLASH_CKPT' not in knobs and 'ckpt_every' in meta:
        knobs['FLASH_CKPT'] = meta['ckpt_every']   # effective value, from the startup banner
    end = max(r['t_end'] for r in reqs)
    m = {'qtrace': 0, 'source': 'slotlog', 'engine': {'version': meta.get('version'), 'build': meta.get('build')},
         'instance': {'label': a.instance, 'served_name': a.served_name},
         'layout': {'cards': meta.get('cards'), 'kv': env.get('FLASH_KV'), 'kv_bits': knobs.get('FLASH_KV_BITS')},
         'knobs': knobs, 'sizes': {'kv_per_tok': 14300, 'end_state': 209e6, 'ck_state': 118e6},
         'window': {'start_epoch': t0, 'start_iso': datetime.fromtimestamp(t0).astimezone().isoformat(), 'end_ms': ms(end),
                    'pod_start_ms': ms(meta['pod_start']) if 'pod_start' in meta else None},
         't_unit': 'ms', 'median_prefill_tps': statistics.median(tps) if tps else 0,
         'notes': {'timestamp': 'start line time (logged after prefill)', 'starts_without_chat_line': dropped,
                   'chat_without_start': meta['unmatched_chat']}}
    os.makedirs(a.out, exist_ok=True)
    json.dump(m, open(os.path.join(a.out, 'meta.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'requests.jsonl'), 'w') as f:
        for r in reqs:
            f.write(json.dumps(dict(
                timestamp=ms(r['t']), input_length=r['P'], output_length=r['gen'], hash_ids=[], q_id=r['id'], q_t_end=ms(r['t_end']),
                q_slot_rec=r['slot'], q_conv_rec=r['conv'], q_reused_rec=r['reused'], q_source_rec=r['source'], q_pinned_rec=r['pinned'],
                q_prefill_tps=r['prefill_tps'], q_gen_tps=r['gen_tps'], q_think_tokens=r['think'], q_effort=r['effort'],
                q_tool_required=r['tool_required'], q_parent=r['parent'], q_div=r['div'], q_pin_class=r['cls'], q_hint=r['cls'])) + '\n')
    with open(os.path.join(a.out, 'decisions.jsonl'), 'w') as f:
        for d in dec:
            f.write(json.dumps(d) + '\n')
    print(f'{a.instance}: {len(reqs)} requests ({dropped} starts without a chat line), {len(dec)} decisions, '
          f'{sum(r["parent"] >= 0 for r in reqs)} with a parent, window {ms(end) / 3.6e6:.2f} h')


if __name__ == '__main__':
    main()
