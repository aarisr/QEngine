"""[al0-0042] qengine_queue_wait_seconds{class}: enqueue -> prefill start, a cumulative Prometheus histogram per class.
Run: python3 tests/test_metrics_queue_wait.py"""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
_nv = types.ModuleType("pynvml")


def _no_nvml():
    raise RuntimeError("NVML disabled in the mock test")


_nv.nvmlInit = _no_nvml
sys.modules["pynvml"] = _nv

import metrics  # noqa: E402

m = metrics.Metrics(os.path.join(tempfile.mkdtemp(prefix="qe_qw_"), "stats.json"))
fake = [1000.0]
metrics.time.time = lambda: fake[0]


def run(cls, wait):
    j = types.SimpleNamespace(ids=list(range(10)), cls=cls)
    m.queued(j, "x")
    fake[0] += wait
    m.prefill_start(j)


run(0, 0.03)   # <= 0.05
run(0, 0.7)    # <= 1
run(0, 7.0)    # <= 10
run(1, 400.0)  # > 300: +Inf only
p = m.qwait_prometheus()
assert "# TYPE qengine_queue_wait_seconds histogram" in p
assert 'qengine_queue_wait_seconds_bucket{class="interactive",le="0.05"} 1' in p, p
assert 'qengine_queue_wait_seconds_bucket{class="interactive",le="0.5"} 1' in p
assert 'qengine_queue_wait_seconds_bucket{class="interactive",le="1"} 2' in p
assert 'qengine_queue_wait_seconds_bucket{class="interactive",le="10"} 3' in p
assert 'qengine_queue_wait_seconds_bucket{class="interactive",le="+Inf"} 3' in p
assert 'qengine_queue_wait_seconds_count{class="interactive"} 3' in p
assert 'qengine_queue_wait_seconds_sum{class="interactive"} 7.73' in p
assert 'qengine_queue_wait_seconds_bucket{class="background",le="300"} 0' in p
assert 'qengine_queue_wait_seconds_bucket{class="background",le="+Inf"} 1' in p
assert "qengine_queue_wait_seconds_bucket" in m.prometheus(), "wired into /metrics"
# a prefill_start for an unknown job observes nothing
m.prefill_start(types.SimpleNamespace(ids=[], cls=0))
assert 'qengine_queue_wait_seconds_count{class="interactive"} 3' in m.qwait_prometheus()
print("test_metrics_queue_wait ok", flush=True)


def test_metrics_queue_wait():  # pytest: the checks above ran at import
    pass
