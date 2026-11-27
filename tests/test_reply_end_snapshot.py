"""CPU test that successful replies snapshot after trim and before the caller is awakened/released."""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))

import app


class OrderedEvent:
    def __init__(self, events):
        self.events = events
        self.is_set_value = False

    def is_set(self):
        return self.is_set_value

    def set(self):
        self.events.append("done")
        self.is_set_value = True


def test_check_done_snapshots_after_stop_trim_before_done_and_release():
    events = []

    class Engine:
        fns = {"qe_snapshot_reply_end"}

        def trim(self, slot, pos):
            events.append("trim")
            return False

        def snapshot_reply_end(self, slot):
            events.append("snapshot")
            return True

        def release(self, slot):
            events.append("release")

    class Metrics:
        def finished(self, job):
            pass

        def reply_end_snapshot(self, seconds):
            events.append("snapshot_metric")

    job = object.__new__(app.Job)
    job.finished = True
    job.done = OrderedEvent(events)
    job.call_in_block = lambda: False
    job.answer_call = lambda: (None, False)
    job.prio = 0
    job.det = None
    job.ds0 = None
    job.gen = types.SimpleNamespace(
        stopping=False,
        profile=types.SimpleNamespace(is_flash=False),
        engine=Engine(),
        metrics=Metrics(),
        dstats=types.SimpleNamespace(read=lambda slot: None),
        cur_slot=0,
    )
    job.slot = 0
    job.keep = 1
    job.ids = [10, 11]
    job.t0 = 0.0
    job.t1 = 1.0
    job.t_enq = 0.0
    job.reused = 0
    job.src = "live"
    job.session = None
    job.finish = "length"
    job.lp = None
    job.n_out = lambda: 1
    job.client_gone = False
    job.shut = False
    job.cancelled = lambda: False
    job.penalty = False
    job.guarded = False
    job.tc = None
    job.p = {}
    job.cut = False
    job.grammar = None
    job.prof = None

    job.check_done()

    assert events.index("trim") < events.index("snapshot")
    assert events.index("snapshot") < events.index("done") < events.index("release")
    assert events.index("snapshot_metric") < events.index("done")


if __name__ == "__main__":
    test_check_done_snapshots_after_stop_trim_before_done_and_release()
    print("test_reply_end_snapshot: ok")
