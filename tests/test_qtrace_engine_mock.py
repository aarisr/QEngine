"""CPU only. Set H4_MOCK_LIB to the compiled tests/qtrace_engine_mock.cpp."""

import os
import sys
import unittest
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.qtrace.engine_mock import ModelEngine, StartRefused


@unittest.skipUnless(
    os.environ.get("H4_MOCK_LIB"), "set H4_MOCK_LIB to the CPU model bridge"
)
class ModelContract(unittest.TestCase):
    def test_overlap_refusal_and_resumption(self):
        with ModelEngine(
            os.environ["H4_MOCK_LIB"], {"knobs": {"FLASH_MAX_SEQS": 2}}
        ) as e:
            ids = array("i", range(1, 11))
            e.set_clock(10)
            a = e.start_begin(ids, hint=0)
            self.assertEqual(e.start_observation()["slot"], a)
            self.assertFalse(e.prefill_more(3, a))
            self.assertTrue(e.prefill_more(7, a))
            _, reused, actual = e.start_finish(a)
            self.assertEqual((reused, actual), (0, a))
            b = e.start_begin(array("i", [25, 26]))
            self.assertNotEqual(a, b)
            self.assertTrue(e.prefill_more(3, b))
            e.start_finish(b)
            with self.assertRaises(StartRefused):
                e.start_begin(array("i", [30]))
            self.assertEqual(e.start_observation()["slot"], -1)
            e.resume(a)
            e.set_limit(a, 2)
            self.assertEqual(len(e.step()), 2)
            self.assertEqual(e.step(), [])
            e.release(b)
            e.release(a)
            self.assertEqual(e.archive_more(), 1)
            e.set_clock(20)
            self.assertGreaterEqual(e.start_begin(ids), 0)
            self.assertGreater(e.start_observation()["maxm"], 0)


if __name__ == "__main__":
    unittest.main()
