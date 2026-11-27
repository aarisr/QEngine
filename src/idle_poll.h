// [event-poll] 0032: QE_IDLE_POLL_S is now the optional idle heartbeat (default 0: none, no GPU call while idle); the
// events (admission, KV growth, prefill start, every step, pool shrink/regrow) call vram_tick, which keeps its own 200 ms gate.
// [idle-power] the cadence of the periodic VRAM-pressure poll (Engine::mem_guard -> vram_tick: cudaSetDevice +
// cudaMemGetInfo on every arm card, NVML below the low mark or every 10 s) while the engine is idle. QE_IDLE_POLL_S
// seconds between polls when nothing runs (default 30; 0: every call, the old behaviour). Work never waits on it:
// the step and prefill paths call vram_tick themselves, and a shrunk pool keeps the full cadence for its way back.
#pragma once
namespace qe_idle_poll {
// true when the idle poll should run now; *last is the time of the previous run (updated when due)
inline bool due(bool idle, bool shrunk, double now, double* last, double every_s) {
    if (!idle || shrunk) return true;
    if (every_s <= 0) return false;  // no heartbeat: an idle engine makes no GPU call; the next event polls
    if (*last > 0 && now - *last < every_s) return false;
    *last = now;
    return true;
}
}  // namespace qe_idle_poll
