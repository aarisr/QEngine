"""CPU test of the archive restore counters in server/metrics.py ([al0-restore-stats]).

The scrape exports per-source restore counters; their sums preserve the old totals, and
libraries without qe_restore_stats_by_source omit the series.
"""
import os, sys, types, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
fake = types.ModuleType("pynvml")
fake.nvmlInit = lambda: None
fake.nvmlDeviceGetCount = lambda: 0
sys.modules.setdefault("pynvml", fake)
from metrics import Metrics

m = Metrics(os.path.join(tempfile.mkdtemp(), "m.json"))
m.engine = {"pool_tokens": 1000, "slots": [], "archives": {
    "tiers": [], "restore_hits": 3, "restore_tokens": 8124,
    "restore_hits_own": 2, "restore_tokens_own": 5000,
    "restore_hits_sibling": 1, "restore_tokens_sibling": 3124,
}}
prom = m.prometheus()
assert "# TYPE qengine_archive_restore_hits_total counter" in prom, prom
assert 'qengine_archive_restore_hits_total{source="own"} 2' in prom
assert 'qengine_archive_restore_hits_total{source="sibling"} 1' in prom
assert 'qengine_archive_restore_tokens_total{source="own"} 5000' in prom
assert 'qengine_archive_restore_tokens_total{source="sibling"} 3124' in prom
assert sum(int(line.rsplit(" ", 1)[1]) for line in prom.splitlines()
           if line.startswith("qengine_archive_restore_hits_total{")) == m.engine["archives"]["restore_hits"]
assert sum(int(line.rsplit(" ", 1)[1]) for line in prom.splitlines()
           if line.startswith("qengine_archive_restore_tokens_total{")) == m.engine["archives"]["restore_tokens"]

# A library without qe_restore_stats_by_source reports no keys: never emit fake zero series.
m.engine = {"pool_tokens": 1000, "slots": [], "archives": {"tiers": []}}
prom = m.prometheus()
assert "qengine_archive_restore_hits_total" not in prom
assert "qengine_archive_restore_tokens_total" not in prom
print("test_metrics_restore_stats ok")
