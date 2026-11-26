// CPU-only observation bridge for tools/qtrace/engine_mock.py; not a production
// qe_* ABI.
#include "qcp_model.h"
#include <memory>
#include <utility>

namespace {
struct Model {
  std::unique_ptr<qcp::Policy> policy;
  qm::Engine engine;
  std::vector<int> scratch;
  size_t seen = 0;
  Model(std::unique_ptr<qcp::Policy> p, qm::Sizes sz, qm::Knobs kb)
      : policy(std::move(p)), engine(*policy, sz, kb) {}
};
} // namespace

extern "C" {
// Fixed POD layouts shared with ctypes. No STL objects or C++ exceptions cross
// the boundary.
struct H4Config {
  double kv_per_tok, end_state, ck_state, archive, pin_budget;
  int slots, ckpt_every, ckpt_min, end_cut, ckpt_keep, kv_stream;
};
struct H4Decision {
  int source, L, maxm, slot, conv, new_conv, victim_slot, victim_conv,
      victim_lost, state_kind;
};
struct H4Degrade {
  double t;
  int action, conv, tokens, resident;
};

Model *h4_open(const char *name, const H4Config *cfg) noexcept {
  try {
    auto policy = qcp::make(name);
    if (!policy || cfg->slots <= 0)
      return nullptr;
    qm::Sizes sz;
    sz.kv_per_tok = cfg->kv_per_tok;
    sz.end_state = cfg->end_state;
    sz.ck_state = cfg->ck_state;
    sz.archive = cfg->archive;
    sz.pin_budget = cfg->pin_budget;
    qm::Knobs kb;
    kb.slots = cfg->slots;
    kb.ckpt_every = cfg->ckpt_every;
    kb.ckpt_min = cfg->ckpt_min;
    kb.end_cut = cfg->end_cut;
    kb.ckpt_keep = cfg->ckpt_keep;
    kb.kv_stream = cfg->kv_stream;
    return new Model(std::move(policy), sz, kb);
  } catch (...) {
    return nullptr;
  }
}
void h4_close(Model *m) noexcept { delete m; }
int h4_start(Model *m, const int *ids, int n, int hint, double t,
             H4Decision *result) noexcept {
  try {
    m->scratch.assign(ids, ids + n);
    const auto o = m->engine.start(m->scratch, hint, 0, 1, t);
    *result = {o.source,      o.L,         o.maxm,        o.slot,
               o.conv,        o.new_conv,  o.victim_slot, o.victim_conv,
               o.victim_lost, o.state_kind};
    return o.refused ? 1 : 0;
  } catch (...) {
    return -1;
  }
}
int h4_finish(Model *m, int slot, const int *ids, int n, double t) noexcept {
  try {
    m->scratch.assign(ids, ids + n);
    m->engine.finish(slot, m->scratch, t);
    return 0;
  } catch (...) {
    return -1;
  }
}
int h4_background(Model *m, double t) noexcept {
  try {
    m->engine.background(t);
    return 0;
  } catch (...) {
    return -1;
  }
}
int h4_next_degrade(Model *m, H4Degrade *out) noexcept {
  if (m->seen == m->engine.degr.size())
    return 0;
  const auto &d = m->engine.degr[m->seen++];
  *out = {d.t, d.action, d.conv, d.tokens, d.resident};
  return 1;
}
}
