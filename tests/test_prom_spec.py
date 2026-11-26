"""/metrics speculative-decoding, cache and thinking series (patch 0028): a fake engine library on top of the mock
engine answers qe_draft_stats (each call adds {2 steps, MTP 5/3, lookup 1/1}) and qe_adapt_stats ({1..8}); a request
must add its draft counter delta, its prompt / cached / generated tokens and its cache source to /metrics.
Run: python3 tests/test_prom_spec.py (CPU only, the mock engine of test_server_mock.py)"""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))

_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the mock test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv

FLASH = os.environ.get("QE_MOCK_MODEL", os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_prom_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"
import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

FAKE = r"""
static long long calls;
void qe_draft_stats(void* e, int slot, long long* out) {
    calls++;
    out[0] = 2 * calls; out[1] = 5 * calls; out[2] = 3 * calls; out[3] = calls; out[4] = calls;
}
void qe_adapt_stats(void* e, long long* out) { for (int i = 0; i < 8; i++) out[i] = i + 1; }
"""
fake = os.path.join(TMP, "fake.c")
open(fake, "w").write(FAKE)
so = os.path.join(TMP, "mock.so")
subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c"), fake])
script = os.path.join(TMP, "script.txt")
args = app.parse_args(["--model", FLASH, "--engine-lib", so, "--metrics-port", "0",
                       "--stats-file", os.path.join(TMP, "stats.json")])
tok, _, _ = args.profile.tokenizer()
with open(script, "w") as f:
    f.write(",".join(map(str, tok.encode("ok\n</think>\n\nDone.<|im_end|>", add_special_tokens=False).ids)) + "\n")
os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, os.path.join(TMP, "log.txt")
gen = app.Generator(args)
c = TestClient(app.build_app(gen, args))
name = c.get("/v1/models").json()["data"][0]["id"]


def prom():
    out = {}
    for ln in gen.metrics.prometheus().splitlines():
        if ln and not ln.startswith("#"):
            k, v = ln.rsplit(" ", 1)
            out[k] = float(v)
    return out


def chat():
    with contextlib.redirect_stdout(io.StringIO()):
        r = c.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text


before = prom()
for k in ("qengine_spec_verify_steps_total", "qengine_spec_drafted_tokens_total", "qengine_spec_accepted_tokens_total",
          "qengine_thinking_tokens_total", "qengine_prompt_cache_requests_total{source=\"live\"}",
          "qengine_prompt_cache_requests_total{source=\"archive\"}",
          "qengine_prompt_cache_requests_total{source=\"other_slot\"}",
          "qengine_prompt_cache_requests_total{source=\"none\"}"):
    assert before[k] == 0, (k, before[k])
for i, k in enumerate(("dfa_verifications", "dfa_drafts_verified", "dfa_drafts_accepted", "mba_verifications",
                       "mba_drafts_verified", "mba_drafts_accepted", "mba_even_split_drafts", "mba_plans")):
    assert before[f"qengine_{k}_total"] == i + 1, k
assert not any("session" in k for k in before), "no session label"

chat()
a = prom()
assert a["qengine_spec_verify_steps_total"] == 2, a
assert a["qengine_spec_drafted_tokens_total"] == 6, a  # MTP 5 + lookup 1
assert a["qengine_spec_accepted_tokens_total"] == 4, a  # MTP 3 + lookup 1
assert a["qengine_requests_total"] == 1 and a["qengine_prompt_tokens_total"] > 0 and a["qengine_generated_tokens_total"] > 0, a
assert a['qengine_prompt_cache_requests_total{source="none"}'] == 1, a
assert a["qengine_thinking_tokens_total"] >= 0
chat()  # the same prompt again: served from the slot (cache source live)
b = prom()
assert b["qengine_spec_verify_steps_total"] == 4 and b["qengine_spec_accepted_tokens_total"] == 8, b
assert sum(b[f'qengine_prompt_cache_requests_total{{source="{s}"}}'] for s in ("live", "archive", "other_slot", "none")) == 2, b
assert b["qengine_cached_tokens_total"] >= a["qengine_cached_tokens_total"]

# the series keep their TYPE lines, and a library without qe_adapt_stats exports no adaptation series
text = gen.metrics.prometheus()
assert "# TYPE qengine_spec_drafted_tokens_total counter" in text and "# TYPE qengine_prompt_cache_requests_total counter" in text
gen.metrics.adapt = None
assert "qengine_mba_plans_total" not in gen.metrics.prometheus()
print("  /metrics: spec verify / drafted / accepted, cache source, thinking and adaptation counters follow the requests")
print("ok")
