"""CPU test of the sibling-rescan metrics in server/metrics.py ([al0-0039]).

Engine::sibling_stats (C API qe_sibling_stats) counts the sibling arms' read-only archives currently indexed and the
idle rescans of QE_DISK_CACHE_READ_DIRS; the scrape exports a gauge and a counter. A library without the call
reports no keys: both are absent, never a fake 0.
"""
import os, sys, types, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
fake = types.ModuleType("pynvml")
fake.nvmlInit = lambda: None
fake.nvmlDeviceGetCount = lambda: 0
sys.modules.setdefault("pynvml", fake)
from metrics import Metrics

m = Metrics(os.path.join(tempfile.mkdtemp(), "m.json"))
m.engine = {"pool_tokens": 1000, "slots": [], "archives": {"tiers": [], "sibling_indexed": 7, "sibling_rescans": 12}}
prom = m.prometheus()
assert "# TYPE qengine_archive_sibling_indexed gauge" in prom, prom
assert "qengine_archive_sibling_indexed 7" in prom
assert "# TYPE qengine_archive_sibling_rescans_total counter" in prom, prom
assert "qengine_archive_sibling_rescans_total 12" in prom

m.engine = {"pool_tokens": 1000, "slots": [], "archives": {"tiers": []}}
prom = m.prometheus()
assert "qengine_archive_sibling_indexed" not in prom
assert "qengine_archive_sibling_rescans_total" not in prom
print("test_metrics_sibling_rescan ok")
