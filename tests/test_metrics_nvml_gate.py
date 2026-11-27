"""CPU test of the scrape-time NVML gate in server/metrics.py ([event-poll] 0032): no NVML while idle, ttl while busy."""
import os, sys, types, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
calls = {"n": 0}
fake = types.ModuleType("pynvml")
fake.nvmlInit = lambda: None
fake.nvmlDeviceGetCount = lambda: (calls.__setitem__("n", calls["n"] + 1), 1)[1]
fake.nvmlDeviceGetHandleByIndex = lambda i: i
fake.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=1, total=2)
fake.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=0)
fake.nvmlDeviceGetPowerUsage = lambda h: 20000
fake.nvmlDeviceGetTotalEnergyConsumption = lambda h: 1000
fake.nvmlDeviceGetEnforcedPowerLimit = lambda h: 100000
fake.nvmlDeviceGetTemperature = lambda h, k: 30
fake.nvmlDeviceGetClockInfo = lambda h, k: 200
sys.modules["pynvml"] = fake
from metrics import Metrics

D = Metrics.nvml_due
assert D(False, False, 1, 0, 5) and D(False, True, 1, 0, 5)      # first scrape reads
assert not D(True, False, 999, 1, 5)                             # idle: cache, however old
assert not D(True, True, 4.9, 0.1, 5) and D(True, True, 5.1, 0.1, 5)
assert D(True, True, 0.2, 0.1, 0)                                # ttl 0: every scrape while busy

m = Metrics(os.path.join(tempfile.mkdtemp(), "m.json"))
n0 = calls["n"]                                                  # (the 1 Hz loop never touches NVML)
m.sample_gpus(100.0)
assert calls["n"] == n0 + 1 and m.gpus and m._power == 20.0
m.sample_gpus(500.0); m.snapshot(); m.snapshot()
assert calls["n"] == n0 + 1                                      # idle: served from cache
m.jobs[1] = {"state": "decode"}
m.sample_gpus(501.0)
assert calls["n"] == n0 + 2                                      # busy and older than 5 s: reads
m.sample_gpus(503.0)
assert calls["n"] == n0 + 2                                      # busy within ttl: cached
print("test_metrics_nvml_gate ok")
