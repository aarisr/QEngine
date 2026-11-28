"""CPU test of the archive restore counters in server/metrics.py ([al0-restore-stats]).

The engine counts every archive restored from the RAM/NVMe store (Engine::restore_stats,
C API qe_restore_stats); the scrape exports them as counters so a restart's restore benefit is
visible in Prometheus instead of only in the "disk cache restore hit" log line.
"""
import os, sys, types, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
fake = types.ModuleType("pynvml")
fake.nvmlInit = lambda: None
fake.nvmlDeviceGetCount = lambda: 0
sys.modules.setdefault("pynvml", fake)
from metrics import Metrics

m = Metrics(os.path.join(tempfile.mkdtemp(), "m.json"))
m.engine = {"pool_tokens": 1000, "slots": [], "archives": {"tiers": [], "restore_hits": 3, "restore_tokens": 8124}}
prom = m.prometheus()
assert "# TYPE qengine_archive_restore_hits_total counter" in prom, prom
assert "qengine_archive_restore_hits_total 3" in prom
assert "qengine_archive_restore_tokens_total 8124" in prom

# A library without qe_restore_stats reports no keys: the counters are absent, never a fake 0.
m.engine = {"pool_tokens": 1000, "slots": [], "archives": {"tiers": []}}
prom = m.prometheus()
assert "qengine_archive_restore_hits_total" not in prom
assert "qengine_archive_restore_tokens_total" not in prom
print("test_metrics_restore_stats ok")
