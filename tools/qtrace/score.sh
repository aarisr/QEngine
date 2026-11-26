#!/usr/bin/env bash
# score.sh: slot logs -> qtrace dirs -> replay under each policy -> spike/results/replay-<inst>-<policy>.json + replay.md
# env: SPIKE (required: the spike dir with traces/raw/*.log), BIN (replay binary; built once under the shared build lock if unset)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd); repo=$(cd "$here/../.." && pwd)
SPIKE=${SPIKE:?set SPIKE to the spike directory (traces/raw/*.log)}
LOCK=${LOCK:-$SPIKE/../.build.lock}
BIN=${BIN:-${TMPDIR:-/tmp}/qcp_replay}
res=$SPIKE/results; mkdir -p "$res"
python3 -I "$here/slotlog_to_qtrace.py" "$SPIKE/traces/raw/qwen38-flash-next-tp4.log" "$SPIKE/traces/A" --instance A
python3 -I "$here/slotlog_to_qtrace.py" "$SPIKE/traces/raw/qwen38-flash-next-b-tp4.log" "$SPIKE/traces/B" --instance B
flock "$LOCK" g++ -std=c++20 -O2 -I "$repo/src" -I "$repo/tests" "$repo/tests/qcp_replay.cpp" -o "$BIN"
pols="default v2"; [ -f "$repo/tests/qcp_frozen_136.h" ] && pols="$pols frozen"
for i in A B; do for p in $pols; do
  "$BIN" "$SPIKE/traces/$i" --policy "$p" --out-decisions "$res/decisions-$i-$p.jsonl" > "$res/replay-$i-$p.json"
done; done
python3 -I - "$res" $pols <<'PY'
import json, sys
res, pols = sys.argv[1], sys.argv[2:]
rows = ['| inst | policy | re-prefill GPU-s/h | saved vs default | recorded re-prefill GPU-s/h | label agreement | exact-L agreement | slot evictions/h | lost/h | archive peak/mean GB |',
        '|---|---|---|---|---|---|---|---|---|---|']
for i in 'AB':
    base = json.load(open(f'{res}/replay-{i}-default.json'))['reprefill_gpu_s_per_h']
    for p in pols:
        r = json.load(open(f'{res}/replay-{i}-{p}.json')); f = r['fidelity']; g = r['reprefill_gpu_s_per_h']
        d = p == 'default'
        c = [i, p, f'{g:.0f}', '-' if d else f'{(1 - g / base) * 100:.0f}% ({base - g:.0f})',
             f"{f['recorded_reprefill_gpu_s_per_h']:.0f}" if d else '-',
             f"{f['label_agreement'] * 100:.1f}%" if d else '-', f"{f['exact_L_agreement'] * 100:.1f}%" if d else '-',
             f"{r['slot_evictions_per_h']:.0f}", f"{r['lost_per_h']:.0f}", f"{r['archive_peak_gb']:.1f} / {r['archive_mean_gb']:.1f}"]
        rows.append('| ' + ' | '.join(c) + ' |')
open(f'{res}/replay.md', 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows))
PY
