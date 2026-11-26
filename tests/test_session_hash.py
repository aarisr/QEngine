"""x-session-hash (gpu-scheduler, MR !67): the header validated and carried into the per-request
record — the [chat] log line and the /api/stats recent[] entry. Present, absent and invalid values.
Run: python3 tests/test_session_hash.py (CPU only, the mock engine of test_server_mock.py)"""
import contextlib
import io
import os
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

FLASH = os.environ.get("QE_MOCK_MODEL", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Qwen3.8-Flash-Next-QE-INT4"))
TMP = tempfile.mkdtemp(prefix="qe_ses_")
os.environ["QE_DIAG_LOG"] = os.path.join(TMP, "diag.log")
os.environ["QE_WARMUP"] = "0"

import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# ---------------------------------------------------------------- the header value itself
H = "0123456789ab"


def hdr(v):
    return {"x-session-hash": v}.get


assert app.session_hash_of(hdr(None)) is None, "absent header: no session"
assert app.session_hash_of(hdr("")) is None, "empty header: no session"
assert app.session_hash_of(hdr("   ")) is None, "whitespace: no session"
assert app.session_hash_of(hdr("ABCDEF012345")) is None, "uppercase is not [0-9a-f]: no session"
assert app.session_hash_of(hdr("0123456789ab-")) is None, "non-hex character: no session"
assert app.session_hash_of(hdr("g" * 12)) is None, "non-hex: no session"
assert app.session_hash_of(hdr("a" * 33)) is None, "longer than 32: no session"
assert app.session_hash_of(hdr(" 0123456789ab")) is None, "leading space: no session"
assert app.session_hash_of(hdr("0123456789ab ")) is None, "trailing space: no session"
assert app.session_hash_of(hdr("0")) == "0", "one character"
assert app.session_hash_of(hdr("a" * 32)) == "a" * 32, "32 characters: the whole value"
assert app.session_hash_of(hdr(H)) == H, "the scheduler's 12-hex prefix"

# ---------------------------------------------------------------- end to end: the [chat] line and /api/stats recent[]
so = os.path.join(TMP, "mock.so")
subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "mock_engine.c")])
log, script = os.path.join(TMP, "log.txt"), os.path.join(TMP, "script.txt")
args = app.parse_args(["--model", FLASH, "--engine-lib", so, "--metrics-port", "0",
                       "--stats-file", os.path.join(TMP, "stats.json")])
tok, _, _ = args.profile.tokenizer()
with open(script, "w") as f:
    f.write(",".join(map(str, tok.encode("ok\n</think>\n\nDone.<|im_end|>", add_special_tokens=False).ids)) + "\n")
os.environ["QE_MOCK_SCRIPT"], os.environ["QE_MOCK_LOG"] = script, log
gen = app.Generator(args)
c = TestClient(app.build_app(gen, args))
name = c.get("/v1/models").json()["data"][0]["id"]


def chat(headers):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        r = c.post("/v1/chat/completions", json={"model": name,
                                                 "messages": [{"role": "user", "content": "hi"}]}, headers=headers)
    assert r.status_code == 200, r.text
    lines = [l for l in out.getvalue().splitlines() if l.startswith("[chat] ")]
    assert len(lines) == 1, lines
    return lines[0]


line = chat({"x-session-hash": H})
assert line.endswith(" | session " + H), line
st = gen.metrics.snapshot()["recent"][0]
assert st.get("session") == H, st

line = chat({})
assert "session" not in line, line
assert "session" not in gen.metrics.snapshot()["recent"][0], "absent header: the record is as before the header"

line = chat({"x-session-hash": "ABCDEF012345"})  # invalid (uppercase): ignored, as absent
assert "session" not in line, line
assert "session" not in gen.metrics.snapshot()["recent"][0], "invalid header: the record is as absent"

print("  x-session-hash: the header validates, the [chat] line and the /api/stats recent[] entry carry it, absent and invalid values change nothing")
print("ok")
