"""x-cache-ttl classes (cache_ttl_of) and the per-tier archive series in /metrics (CPU only).
Run: python3 tests/test_cache_session.py"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
_nv = types.ModuleType("pynvml")
sys.modules.setdefault("pynvml", _nv)

import app  # noqa: E402

for k in ("QE_CACHE_TTL_SHORT_S", "QE_CACHE_TTL_DEFAULT_S", "QE_CACHE_TTL_LONG_S", "QE_CACHE_MAX_AGE_S"):
    os.environ.pop(k, None)
h = lambda v: (lambda name: v if name == "x-cache-ttl" else None)
assert app.cache_ttl_of(h(None)) == 86400
assert app.cache_ttl_of(h("short")) == 3600 and app.cache_ttl_of(h(" LONG ")) == 7 * 86400
assert app.cache_ttl_of(h("bogus")) == 86400
os.environ["QE_CACHE_TTL_LONG_S"] = "99999999"
assert app.cache_ttl_of(h("long")) == 7 * 86400  # capped at the maximum age
os.environ["QE_CACHE_TTL_SHORT_S"] = "5"
assert app.cache_ttl_of(h("short")) == 5
assert app.SESSION_HASH.match("0123456789ab") and not app.SESSION_HASH.match("../x")
print("test_cache_session: ok")
