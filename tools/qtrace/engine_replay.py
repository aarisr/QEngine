"""Drive an observable, engine.py-shaped backend; write qcp_replay's exact D wire.

CLI supports the instrumented CPU mock only. The stock GPU ABI lacks full D
observations; never substitute CPU predictions for a missing engine witness.
"""

import argparse
import json
import math
import sys
from array import array
from itertools import islice, zip_longest
from pathlib import Path

if __package__:
    from .engine_mock import ModelEngine
else:
    from engine_mock import ModelEngine


def load_window(trace_dir, limit):
    if limit < 1:
        raise ValueError("limit must be positive")
    meta = json.loads((trace_dir / "meta.json").read_bytes())
    with (trace_dir / "requests.jsonl").open("rb") as f:
        rows = [json.loads(line) for line in islice(f, limit)]
    if len(rows) != limit:
        raise ValueError(f"expected {limit} requests, got {len(rows)}")
    return meta, rows


def compare_decisions(reference, actual):
    with reference.open("rb") as left, actual.open("rb") as right:
        count = 0
        for count, (expected, got) in enumerate(zip_longest(left, right), 1):
            if expected != got:
                raise ValueError(
                    f"D mismatch at record {count}: expected {expected!r}, got {got!r}"
                )
        if not count:
            raise ValueError("empty decision stream")
        return count


def _emit(out, fields):
    # Every t in this wire is a numeric literal with exactly three decimals, not a JSON string.
    line = ",".join(
        '"' + k + '":' + (format(v, ".3f") if k == "t" else json.dumps(v))
        for k, v in fields.items()
    )
    out.write(("{" + line + "}\n").encode("ascii"))


def replay(engine, rows, out, *, prefill_budget=4096):
    """Run one fresh arm; retain parent prompts only while an unstarted child needs them."""
    if not rows or prefill_budget <= 0:
        raise ValueError("nonempty window and positive prefill budget required")
    events, children = [], [0] * len(rows)
    for i, r in enumerate(rows):
        t, end = r["timestamp"], r["q_t_end"]
        parent = r.get("q_parent", -1)
        if not (math.isfinite(t) and math.isfinite(end) and 0 <= t < end):
            raise ValueError(f"request {i}: invalid arrival/finish interval")
        if min(r["input_length"], r["output_length"]) < 1:
            raise ValueError(
                f"request {i}: engine requires a prompt and emits a first token"
            )
        if not -1 <= parent < i or (parent >= 0 and rows[parent]["timestamp"] > t):
            raise ValueError(f"request {i}: invalid or not-yet-started parent")
        if min(r.get("q_div", 0), r.get("q_pin_class", 0)) < 0:
            raise ValueError(f"request {i}: negative prefix length")
        if parent >= 0:
            children[parent] += 1
        events.extend(((end, 0, i), (t, 1, i)))
    prompts, classes, slots = {}, {}, {}
    fresh, tick = 1, 1000

    def degrade(req):
        for d in engine.degradations():
            _emit(
                out,
                {
                    "seam": "S2",
                    "kind": "degrade",
                    "t": d["t"],
                    "req": req,
                    "action": d["action"],
                    "conv": d["conv"],
                    "tokens": d["tokens"],
                    "resident": d["resident"],
                },
            )

    def archive(t):
        engine.set_clock(t)
        while True:
            rc = engine.archive_more(64 << 20)
            if rc in (1, 2):  # done, or deferred until a later pass
                return
            if rc != 0:
                raise RuntimeError(f"unexpected archive_more status {rc}")

    def background_until(t, req):
        nonlocal tick
        while tick <= t:
            archive(tick)
            tick += 1000
        degrade(req)

    for t, arrival, i in sorted(events):
        background_until(t, i)
        engine.set_clock(t)
        r = rows[i]
        if not arrival:
            slot = slots.pop(i, None)
            if slot is None:  # refused at start
                continue
            remaining = r["output_length"] - 1
            engine.resume(slot)
            engine.set_limit(slot, remaining)
            while remaining:
                got = len(engine.step())
                if not 0 < got <= remaining:
                    raise RuntimeError(
                        f"request {i}: decode returned {got} tokens for limit {remaining}"
                    )
                remaining -= got
            engine.release(slot)
            fresh += r[
                "output_length"
            ]  # reference allocates distinct generated IDs on finish
            continue

        parent, pin, n = (
            r.get("q_parent", -1),
            r.get("q_pin_class", 0),
            r["input_length"],
        )
        if parent >= 0:
            ids = prompts[parent][: min(n, r.get("q_div", 0))]
            children[parent] -= 1
            if not children[parent]:
                del prompts[parent]
        elif pin:
            ordinal = classes.setdefault(pin, len(classes))
            start = 1_000_000_000 + ordinal * 1_000_000
            if min(pin, n) > 1_000_000 or start + min(pin, n) > 2_000_000_000:
                raise ValueError("pin-class synthetic ID ranges would overlap")
            ids = array("i", range(start, start + min(pin, n)))
        else:
            ids = array("i")
        end = fresh + n - len(ids)
        if end > 1_000_000_000:
            raise ValueError("fresh and pin-class synthetic ID ranges would overlap")
        ids.extend(range(fresh, end))
        fresh = end
        if children[i]:
            prompts[i] = ids

        archive(t)
        degrade(i)
        try:
            slot = engine.start_begin(ids, hint=r.get("q_hint", 0))
        except engine.refusal:
            slot = None
        d = engine.start_observation()
        degrade(i)
        if slot is not None:
            if d["slot"] != slot:
                raise RuntimeError(f"request {i}: observed a different slot")
            while not engine.prefill_more(prefill_budget, slot):
                pass
            _, reused, actual = engine.start_finish(slot)
            if (actual, reused) != (slot, d["L"]):
                raise RuntimeError(
                    f"request {i}: prefill result disagrees with observed decision"
                )
            slots[i] = slot
        fields = (
            "source",
            "L",
            "maxm",
            "slot",
            "conv",
            "new_conv",
            "victim_slot",
            "victim_conv",
            "victim_lost",
            "state_kind",
        )
        _emit(out, {"seam": "S7", "req": i, "t": t, **{key: d[key] for key in fields}})
    background_until(rows[-1]["q_t_end"], len(rows) - 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace_dir", type=Path)
    p.add_argument(
        "--mock-lib",
        type=Path,
        required=True,
        help="compiled tests/qtrace_engine_mock.cpp (CPU only)",
    )
    p.add_argument("--out-decisions", type=Path, required=True)
    p.add_argument(
        "--compare", type=Path, help="independent qcp_replay output on the same window"
    )
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--policy", choices=("default", "v2"), default="default")
    args = p.parse_args()
    if args.compare and args.compare.resolve() == args.out_decisions.resolve():
        p.error("reference and output must be distinct files")
    meta, rows = load_window(args.trace_dir, args.limit)
    with (
        ModelEngine(args.mock_lib, meta, args.policy) as engine,
        args.out_decisions.open("xb") as out,
    ):
        replay(engine, rows, out)
    if args.compare:
        print(
            f"CPU mock: {compare_decisions(args.compare, args.out_decisions)} D records byte-identical ({len(rows)} requests)"
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        sys.exit(str(exc))
