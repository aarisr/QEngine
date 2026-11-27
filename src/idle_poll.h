// [idle-power] the cadence of the periodic VRAM-pressure poll (Engine::mem_guard -> vram_tick: cudaSetDevice +
// cudaMemGetInfo on every arm card, NVML below the low mark or every 10 s) while the engine is idle. QE_IDLE_POLL_S
// seconds between polls when nothing runs (default 30; 0: every call, the old behaviour). Work never waits on it:
// the step and prefill paths call vram_tick themselves, and a shrunk pool keeps the full cadence for its way back.
#pragma once
namespace qe_idle_poll {
// true when the idle poll should run now; *last is the time of the previous run (updated when due)
inline bool due(bool idle, bool shrunk, double now, double* last, double every_s) {
    if (!idle || shrunk || every_s <= 0) return true;
    if (*last > 0 && now - *last < every_s) return false;
    *last = now;
    return true;
}
}  // namespace qe_idle_poll
