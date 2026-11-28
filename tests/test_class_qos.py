"""Class QoS (x-request-class): the request class from the header, the scheduler's class order in
pick_next (interactive before background, aging still inside a class), and the metrics view
(queued_by_class snapshot counts, class on jobs[], Prometheus gauges). Run: python3 tests/test_class_qos.py"""
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

import app  # noqa: E402
import metrics  # noqa: E402

# ---------------------------------------------------------------- x-request-class header
assert app.CLS_BG == {"background"}


def hdr(v):
    return {"x-request-class": v}.get


assert app.request_class_of(hdr(None)) == 0, "missing header: interactive"
assert app.request_class_of(hdr("foreground")) == 0, "unknown value: interactive"
assert app.request_class_of(hdr("backgroundx")) == 0, "not a full match: interactive"
assert app.request_class_of(hdr("")) == 0, "empty: interactive"
assert app.request_class_of(hdr("   ")) == 0, "whitespace only: interactive"
assert app.request_class_of(hdr("background")) == 1
assert app.request_class_of(hdr("Background")) == 1, "case-insensitive"
assert app.request_class_of(hdr("BACKGROUND")) == 1
assert app.request_class_of(hdr("  background  ")) == 1, "stripped"
assert app.request_class_of(hdr("\tBackground\n")) == 1

# ---------------------------------------------------------------- pick_next: class first, then aged size
NOW = 1000.0


def job(cls, est, waited=0.0):
    j = types.SimpleNamespace(ids=app.Ids(range(est)), cls=cls, est=est, t_enq=NOW - waited, t_est=NOW, p={})
    return j


def adm_with(queue, aging=4000):
    e = types.SimpleNamespace()
    e.set_steering = lambda *a: None
    e.probe_cached = lambda ids: 0
    g = types.SimpleNamespace(queue=queue, engine=e,
                              args=types.SimpleNamespace(prefill_aging=aging, preempt_min=8192,
                                                         preempt_max=1024, mtp_k=3))
    return app.Admission(g)


# a fresh background request with the least left to compute waits behind an interactive one
g = adm_with([job(1, 1000), job(0, 50000)])
assert g.pick_next(NOW, []) == [1, 0], "background last whatever its estimate"

# and an old, large background request does not overtake a fresh interactive one (aging is per class)
g = adm_with([job(0, 50000), job(1, 50000, waited=100)])
assert g.pick_next(NOW, []) == [0, 1], "class beats age across classes"

# aging still orders inside a class: 5K waiting 2 s (key -3000) before a fresh 1K, both interactive
g = adm_with([job(0, 5000, waited=2), job(0, 1000)])
assert g.pick_next(NOW, []) == [0, 1], "aging within interactive"

# ... and inside background: a 50K that waited 20 s (key -30000) before a fresh 1K, then an interactive one
g = adm_with([job(1, 1000), job(0, 50000), job(1, 50000, waited=20)])
assert g.pick_next(NOW, []) == [1, 2, 0], "interactive first, then aged background before fresh background"

# no aging (0): strict class then size
g = adm_with([job(1, 1000), job(0, 50000)], aging=0)
assert g.pick_next(NOW, []) == [1, 0]
g = adm_with([job(1, 50000), job(1, 1000)], aging=0)
assert g.pick_next(NOW, []) == [1, 0], "background order by size alone without aging"
assert adm_with([]).pick_next(NOW, []) == [], "empty queue"

# ---------------------------------------------------------------- [al0-0040] priority within interactive
assert app.CLS_PRIO == {"priority"}
assert app.request_prio_of(hdr("priority")) == 1
assert app.request_prio_of(hdr("  Priority ")) == 1, "case-insensitive, stripped"
assert app.request_prio_of(hdr("background")) == 0
assert app.request_prio_of(hdr(None)) == 0 and app.request_prio_of(hdr("prio")) == 0
assert app.request_class_of(hdr("priority")) == 0, "priority stays engine class 0 (interactive rules)"


def pjob(cls, est, prio, waited=0.0):
    j = job(cls, est, waited)
    j.prio = prio
    return j


# a fresh, LARGER priority request goes before a smaller interactive one that has waited
g = adm_with([pjob(0, 1000, 0, waited=2), pjob(0, 5000, 1)])
assert g.pick_next(NOW, []) == [1, 0], "priority first within interactive"
# background stays last even behind... nothing: a priority request never jumps the class order upward wrongly
g = adm_with([pjob(1, 10, 0), pjob(0, 50000, 0), pjob(0, 50000, 1)])
assert g.pick_next(NOW, []) == [2, 1, 0], "priority, then interactive, then background"
# aging still orders inside the priority group
g = adm_with([pjob(0, 5000, 1, waited=2), pjob(0, 1000, 1)])
assert g.pick_next(NOW, []) == [0, 1], "aging within priority"
# a job without the attribute (older callers) counts as not priority
legacy = job(0, 10)
g = adm_with([legacy, pjob(0, 10000, 1)])
assert g.pick_next(NOW, []) == [1, 0], "missing prio = 0"
# pause wait: a priority request pauses after PRIO_PAUSE_WAIT, others after pause_wait
g = adm_with([])
assert app.PRIO_PAUSE_WAIT == 0.5
assert g.pause_wait_of(pjob(0, 10, 1)) == min(app.PAUSE_WAIT, app.PRIO_PAUSE_WAIT)
assert g.pause_wait_of(pjob(0, 10, 0)) == app.PAUSE_WAIT
assert g.pause_wait_of(job(0, 10)) == app.PAUSE_WAIT
# [al0-0041] an interactive request pauses at once while background decodes; background callers keep their wait
assert g.pause_wait_of(job(0, 10), True) == 0.0
assert g.pause_wait_of(pjob(0, 10, 1), True) == 0.0
assert g.pause_wait_of(job(1, 10), True) == app.PAUSE_WAIT
assert g.pause_wait_of(job(0, 10), False) == app.PAUSE_WAIT
g.g.metrics = types.SimpleNamespace(active_by_class=lambda: {"interactive": 2, "background": 1})
assert g.background_active() is True
g.g.metrics = types.SimpleNamespace(active_by_class=lambda: {"interactive": 2, "background": 0})
assert g.background_active() is False
g.g.metrics = None
assert g.background_active() is False

# ---------------------------------------------------------------- metrics: class on jobs, queued_by_class, gauges
tmp = tempfile.mkdtemp(prefix="qe_cls_")
m = metrics.Metrics(os.path.join(tmp, "stats.json"))

jb, ji = types.SimpleNamespace(ids=list(range(100)), cls=1), types.SimpleNamespace(ids=list(range(50)), cls=0)
m.queued(jb, "bg agent")
m.queued(ji, "user")
jx = types.SimpleNamespace(ids=list(range(200)), cls=2)  # a class the engine does not know: interactive
m.queued(jx, "odd", cls=None)

s = m.snapshot()
assert s["queued_by_class"] == {"interactive": 2, "background": 1}, s["queued_by_class"]
by_label = {j["label"]: j for j in s["jobs"]}
assert by_label["bg agent"]["class"] == "background" and by_label["bg agent"]["state"] == "queued"
assert by_label["user"]["class"] == "interactive"
assert by_label["odd"]["class"] == "interactive", "only cls 1 is background"

p = m.prometheus()
assert 'qengine_queued_by_class{class="interactive"} 2' in p, p
assert 'qengine_queued_by_class{class="background"} 1' in p
assert "# TYPE qengine_queued_by_class gauge" in p
assert "qengine_queued_requests 3" in p
assert "qengine_active_sequences 0" in p

# a background request admitted leaves the background bucket, stays in jobs[] with its class
m.prefill_start(jb)
s = m.snapshot()
assert s["queued_by_class"] == {"interactive": 2, "background": 0}, s["queued_by_class"]
by_label = {j["label"]: j for j in s["jobs"]}
assert by_label["bg agent"]["state"] == "prefill" and by_label["bg agent"]["class"] == "background"
assert "qengine_active_sequences 1" in m.prometheus()

# and it finishes: out of jobs[] and of the class counts
m.prefill_done(types.SimpleNamespace(ids=jb.ids, reused=0, slot=0, src="live"))
m.finished(jb)
s = m.snapshot()
assert s["queued_by_class"] == {"interactive": 2, "background": 0}
assert [j["label"] for j in s["jobs"]] == ["user", "odd"]

print("test_class_qos: ok", flush=True)


def test_class_qos():  # pytest: the checks above ran at import
    pass


if __name__ == "__main__":
    sys.exit(0)
