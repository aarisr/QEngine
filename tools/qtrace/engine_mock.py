"""H4 CPU backend. Decisions come from qm::Engine, never a reference D file.

The lifecycle mirrors server.engine.Engine; set_clock/start_observation/degradations
are the mock's observation hooks. Synthetic generated IDs carry no model semantics.
"""

import ctypes as C
from array import array
from dataclasses import dataclass, field


class StartRefused(RuntimeError):
    """The CPU model refused admission. The refused S7 is still observable."""


class _Config(C.Structure):
    _fields_ = [
        (k, C.c_double)
        for k in ("kv_per_tok", "end_state", "ck_state", "archive", "pin_budget")
    ] + [
        (k, C.c_int)
        for k in (
            "slots",
            "ckpt_every",
            "ckpt_min",
            "end_cut",
            "ckpt_keep",
            "kv_stream",
        )
    ]


class _Decision(C.Structure):
    _fields_ = [
        (k, C.c_int)
        for k in (
            "source",
            "L",
            "maxm",
            "slot",
            "conv",
            "new_conv",
            "victim_slot",
            "victim_conv",
            "victim_lost",
            "state_kind",
        )
    ]


class _Degrade(C.Structure):
    _fields_ = [("t", C.c_double)] + [
        (k, C.c_int) for k in ("action", "conv", "tokens", "resident")
    ]


@dataclass
class _Slot:
    prefill: int
    reused: int
    batch: int
    generated: array = field(default_factory=lambda: array("i"))
    remaining: int = 0


class ModelEngine:
    refusal = StartRefused

    def __init__(self, lib_path, meta, policy="default"):
        self.lib = C.CDLL(str(lib_path))
        handle, num, buf = C.c_void_p, C.c_int, C.c_void_p
        for name, ret, args in (
            ("open", handle, [C.c_char_p, C.POINTER(_Config)]),
            ("close", None, [handle]),
            ("start", num, [handle, buf, num, num, C.c_double, C.POINTER(_Decision)]),
            ("finish", num, [handle, num, buf, num, C.c_double]),
            ("background", num, [handle, C.c_double]),
            ("next_degrade", num, [handle, C.POINTER(_Degrade)]),
        ):
            f = getattr(self.lib, "h4_" + name)
            f.restype, f.argtypes = ret, args
        sizes, k = meta.get("sizes", {}), meta.get("knobs", {})
        cfg = _Config(
            sizes.get("kv_per_tok", 14300),
            sizes.get("end_state", 209e6),
            sizes.get("ck_state", 118e6),
            k.get("FLASH_KV_ARCHIVE_GB", 80) * 1e9,
            k.get("FLASH_PREFIX_GB", 8) * 1e9,
            k.get("FLASH_MAX_SEQS", 16),
            k.get("FLASH_CKPT", 8192),
            k.get("FLASH_CKPT_MIN", 512),
            k.get("FLASH_END_CUT", 4096),
            k.get("FLASH_CKPT_KEEP", 6),
            k.get("FLASH_KV", "ram") == "ram",
        )
        self.h = self.lib.h4_open(policy.encode(), C.byref(cfg))
        if not self.h:
            raise ValueError("CPU model creation failed")
        self.slots = {}
        self.active = None
        self.t = 0.0
        self.last = None
        self.serial = 2**31 - 1  # disjoint from all mock prompt IDs (<2e9)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.lib.h4_close(self.h)
        self.h = None

    def set_clock(self, t):
        if t < self.t:
            raise ValueError("virtual clock went backwards")
        self.t = t

    def start_begin(self, ids, k=3, affinity=0, hint=0, mm=None):
        if mm is not None or affinity or k < 0 or not ids:
            raise ValueError("unsupported start input")
        if not isinstance(ids, array) or ids.typecode != "i":
            ids = array("i", ids)
        if ids.itemsize != C.sizeof(C.c_int):
            raise ValueError("token type must match C int")
        d = _Decision()
        rc = self.lib.h4_start(
            self.h, ids.buffer_info()[0], len(ids), hint, self.t, C.byref(d)
        )
        if rc < 0:
            raise RuntimeError("CPU start failed")
        self.last = d
        if rc:
            raise StartRefused("CPU admission refused")
        self.slots[d.slot] = _Slot(len(ids) - d.L, d.L, k + 1)
        return d.slot

    def start_observation(self):
        d = self.last
        if d is None:
            raise RuntimeError("no observed start")
        return {
            k: (
                bool(getattr(d, k))
                if k in ("new_conv", "victim_lost")
                else getattr(d, k)
            )
            for k in (spec[0] for spec in d._fields_)
        }

    def degradations(self):
        d = _Degrade()
        while self.lib.h4_next_degrade(self.h, C.byref(d)):
            yield {
                "t": d.t,
                "action": ("drop_rows", "drop_spare_states", "drop_copy", "lose")[
                    d.action
                ],
                "conv": d.conv,
                "tokens": d.tokens,
                "resident": bool(d.resident),
            }

    def prefill_more(self, budget, slot=None):
        s = self.slots[slot]
        if budget <= 0 or s.generated:
            raise ValueError("invalid prefill")
        s.prefill = max(0, s.prefill - budget)
        return s.prefill == 0

    def _generate(self, s, count):
        if self.serial - count < 2_000_000_000:
            raise OverflowError("exhausted disjoint synthetic output IDs")
        ids = list(range(self.serial, self.serial - count, -1))
        self.serial -= count
        s.generated.extend(ids)
        return ids

    def start_finish(self, slot=None):
        s = self.slots[slot]
        if s.prefill or s.generated:
            raise ValueError("start_finish outside prefilled state")
        self.active = slot
        return self._generate(s, 1)[0], s.reused, slot

    def resume(self, slot):
        if not self.slots[slot].generated:
            raise ValueError("resume before start_finish")
        self.active = slot

    def set_limit(self, slot, tokens):
        if tokens < 0:
            raise ValueError("negative decode limit")
        self.slots[slot].remaining = tokens

    def step(self):
        s = self.slots[self.active]
        n = min(s.remaining, s.batch)
        s.remaining -= n
        return self._generate(s, n)

    def release(self, slot):
        s = self.slots[slot]
        if not s.generated or s.remaining:
            raise ValueError("release before decode completion")
        if self.lib.h4_finish(
            self.h, slot, s.generated.buffer_info()[0], len(s.generated), self.t
        ):
            raise RuntimeError("CPU finish failed")
        del self.slots[slot]
        if slot == self.active:
            self.active = None

    def archive_more(self, budget=64 << 20) -> int:
        if budget <= 0 or self.lib.h4_background(self.h, self.t):
            raise RuntimeError("CPU archive pass failed")
        return 1
