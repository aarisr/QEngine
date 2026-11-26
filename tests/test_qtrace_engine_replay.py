"""Run with H4_MOCK_LIB, H4_QCP_REPLAY. Reference input never enters the mock."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qtrace.engine_mock import ModelEngine
from tools.qtrace.engine_replay import compare_decisions, replay


class DecisionCompareTests(unittest.TestCase):
    def test_identical_streams_count_records(self):
        with tempfile.TemporaryDirectory() as td:
            left, right = Path(td) / "left.jsonl", Path(td) / "right.jsonl"
            payload = b'{"op":"prefill"}\n{"op":"decode"}\n'
            left.write_bytes(payload)
            right.write_bytes(payload)
            self.assertEqual(compare_decisions(left, right), 2)

    def test_first_mismatch_and_empty_stream_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            left, right = Path(td) / "left.jsonl", Path(td) / "right.jsonl"
            left.write_bytes(b'{"op":"prefill"}\n{"op":"decode"}\n')
            right.write_bytes(b'{"op":"prefill"}\n{"op":"start"}\n')
            with self.assertRaisesRegex(ValueError, "D mismatch at record 2"):
                compare_decisions(left, right)
            right.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty decision stream"):
                compare_decisions(right, right)


class PerturbedEngine(ModelEngine):
    def start_observation(self):
        d = super().start_observation()
        d["maxm"] += 1  # mutate the mock observation, not either output file
        return d


@unittest.skipUnless(
    os.getenv("H4_QCP_REPLAY") and os.getenv("H4_MOCK_LIB"),
    "requires the CPU reference and model bridge",
)
class DecisionWire(unittest.TestCase):
    def test_finish_before_arrival_and_refused_overlap(self):
        rows = [
            dict(
                timestamp=t,
                q_t_end=end,
                input_length=n,
                output_length=4,
                q_parent=p,
                q_div=div,
                q_hint=0,
                q_pin_class=0,
            )
            for t, end, n, p, div in [
                (0, 1000, 32, -1, 0),
                (100, 200, 20, -1, 0),
                (1000, 2000, 40, 0, 31),
            ]
        ]
        meta = {"knobs": {"FLASH_MAX_SEQS": 1}}
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "meta.json").write_text(json.dumps(meta))
            (folder / "requests.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            ref = folder / "reference"
            actual = folder / "actual"
            subprocess.run(
                [os.environ["H4_QCP_REPLAY"], str(folder), "--out-decisions", str(ref)],
                check=True,
                capture_output=True,
            )
            with (
                ModelEngine(os.environ["H4_MOCK_LIB"], meta) as engine,
                actual.open("wb") as out,
            ):
                replay(engine, rows, out, prefill_budget=7)
            compare_decisions(ref, actual)
            starts = [
                d
                for d in map(json.loads, actual.read_bytes().splitlines())
                if d["seam"] == "S7"
            ]
            self.assertEqual(
                [(d["req"], d["slot"]) for d in starts], [(0, 0), (1, -1), (2, 0)]
            )
            with (
                PerturbedEngine(os.environ["H4_MOCK_LIB"], meta) as engine,
                actual.open("wb") as out,
            ):
                replay(engine, rows, out, prefill_budget=7)
            with self.assertRaisesRegex(ValueError, "record 1"):
                compare_decisions(ref, actual)

    def test_archive_defers_to_later_pass(self):
        class Deferred(ModelEngine):
            passes = []

            def archive_more(self, budget=64 << 20):
                self.passes.append(self.t)
                if self.t == 0 and len(self.passes) == 1:
                    return 2
                return super().archive_more(budget)

        with Deferred(os.environ["H4_MOCK_LIB"], {}) as engine:
            replay(
                engine,
                [
                    {
                        "timestamp": 0,
                        "q_t_end": 1000,
                        "input_length": 3,
                        "output_length": 1,
                    }
                ],
                io.BytesIO(),
            )
            self.assertEqual(engine.passes, [0, 1000])


if __name__ == "__main__":
    unittest.main()
