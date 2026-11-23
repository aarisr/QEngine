#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

// CPU-testable QKVA0004 metadata codec. Payload bytes stay in the engine's
// existing per-GPU buffers; this helper defines only the durable index/header
// contract.
namespace qkva {
struct Header {
  std::string fingerprint;
  int32_t n = 0;
  std::vector<int32_t> subs;
  std::vector<uint64_t> sizes, hashes;
  uint32_t pinned = 0;
  int32_t split = 0;
  std::vector<int32_t> toks;
  size_t data_off = 0, hash_off = 0;
};
enum class RamTier { none, ordinary, prefix };
inline RamTier ram_tier(bool pinned, bool resident) {
  return !resident ? RamTier::none
         : pinned  ? RamTier::prefix
                   : RamTier::ordinary;
}

template <class T> inline void put(std::vector<char> &out, const T &v) {
  const char *p = reinterpret_cast<const char *>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}
struct Reader {
  const char *p;
  size_t n, at = 0;
  template <class T> bool take(T &v) {
    if (sizeof(T) > n - at)
      return false;
    memcpy(&v, p + at, sizeof(T));
    at += sizeof(T);
    return true;
  }
  bool bytes(void *dst, size_t len) {
    if (len > n - at)
      return false;
    if (len)
      memcpy(dst, p + at, len);
    at += len;
    return true;
  }
};

inline uint64_t metadata_hash(const Header &h) {
  uint64_t v = 14695981039346656037ull;
  auto mix = [&](const void *p, size_t n) {
    const auto *b = static_cast<const unsigned char *>(p);
    for (size_t i = 0; i < n; ++i) {
      v ^= b[i];
      v *= 1099511628211ull;
    }
  };
  mix(&h.n, sizeof(h.n));
  mix(&h.pinned, sizeof(h.pinned));
  mix(&h.split, sizeof(h.split));
  for (int32_t x : h.subs)
    mix(&x, sizeof(x));
  for (int32_t x : h.toks)
    mix(&x, sizeof(x));
  return v;
}
inline bool valid(const Header &h, size_t ngpu, size_t max_subs) {
  if (h.n <= 0 || h.toks.size() != (size_t)h.n || h.pinned > 1 || h.split < 0 ||
      h.split > h.n || h.sizes.size() != ngpu ||
      h.hashes.size() != ngpu * (1 + h.subs.size()) || h.subs.size() > max_subs)
    return false;
  if (h.pinned && (h.split != 0 || !h.subs.empty()))
    return false;
  if (!h.pinned && h.split == h.n)
    return false;
  int32_t prev = 0;
  for (int32_t x : h.subs) {
    if (x <= prev || x >= h.n)
      return false;
    prev = x;
  }
  return true;
}
inline std::vector<char> encode(const Header &h, size_t *hash_off = nullptr) {
  std::vector<char> out;
  static constexpr char magic[8] = {'Q', 'K', 'V', 'A', '0', '0', '0', '4'};
  out.insert(out.end(), magic, magic + 8);
  put(out, (uint32_t)h.fingerprint.size());
  out.insert(out.end(), h.fingerprint.begin(), h.fingerprint.end());
  put(out, h.n);
  put(out, (uint32_t)h.subs.size());
  for (int32_t x : h.subs)
    put(out, x);
  put(out, (uint32_t)h.sizes.size());
  for (uint64_t x : h.sizes)
    put(out, x);
  put(out, (uint32_t)h.hashes.size());
  if (hash_off)
    *hash_off = out.size();
  for (uint64_t x : h.hashes)
    put(out, x);
  put(out, h.pinned);
  put(out, h.split);
  put(out, metadata_hash(h));
  for (int32_t x : h.toks)
    put(out, x);
  return out;
}
inline bool decode(const char *data, size_t len, const std::string &fingerprint,
                   size_t ngpu, size_t max_subs, Header &out) {
  if (len < 8 || memcmp(data, "QKVA0004", 8))
    return false;
  Reader r{data, len, 8};
  uint32_t fl = 0, ns = 0, ng = 0, nh = 0;
  Header h;
  if (!r.take(fl) || fl > 65536 || fl > r.n - r.at)
    return false;
  h.fingerprint.resize(fl);
  if (!r.bytes(h.fingerprint.data(), fl) || h.fingerprint != fingerprint ||
      !r.take(h.n) || h.n <= 0 || !r.take(ns) || ns > max_subs)
    return false;
  h.subs.resize(ns);
  for (auto &x : h.subs)
    if (!r.take(x))
      return false;
  if (!r.take(ng) || ng != ngpu)
    return false;
  h.sizes.resize(ng);
  for (auto &x : h.sizes)
    if (!r.take(x))
      return false;
  if (!r.take(nh) || nh != ng * (1 + ns))
    return false;
  h.hash_off = r.at;
  h.hashes.resize(nh);
  for (auto &x : h.hashes)
    if (!r.take(x))
      return false;
  uint64_t hh = 0;
  if (!r.take(h.pinned) || !r.take(h.split) || !r.take(hh))
    return false;
  if ((size_t)h.n > (r.n - r.at) / sizeof(int32_t))
    return false;
  h.toks.resize(h.n);
  for (auto &x : h.toks)
    if (!r.take(x))
      return false;
  if (!valid(h, ngpu, max_subs) || metadata_hash(h) != hh)
    return false;
  h.data_off = r.at;
  out = std::move(h);
  return true;
}
inline bool same_prefix(const Header &child, const Header &pin) {
  return pin.pinned && !child.pinned && child.split > 0 && pin.split == 0 &&
         pin.n == child.split && pin.fingerprint == child.fingerprint &&
         pin.toks.size() >= (size_t)child.split &&
         std::equal(pin.toks.begin(), pin.toks.end(), child.toks.begin());
}
// Read + verify one QKVA0004 file header from f (the file starts with the magic; every
// read is bounds-checked against file_size, so a truncated or corrupt file returns false
// and is dropped by the caller - never a crash). The payload bytes are not read here
// (archive_load reads them at out.data_off). This is the QKVA0004 side of
// Engine::disk_init; the CPU test archive_format_unit runs it on real on-disk files.
inline bool decode_file(FILE* f, size_t file_size, const std::string& fingerprint,
                        size_t ngpu, size_t max_subs, int max_n, Header& out) {
  fseek(f, 0, SEEK_SET);
  char magic[8];
  if (fread(magic, 1, sizeof(magic), f) != sizeof(magic))
    return false;
  std::vector<char> raw(magic, magic + sizeof(magic));
  auto rd = [&](auto& v) {
    if (fread(&v, sizeof(v), 1, f) != 1)
      return false;
    const char* p = reinterpret_cast<const char*>(&v);
    raw.insert(raw.end(), p, p + sizeof(v));
    return true;
  };
  uint32_t fl = 0, ns = 0, ng = 0, nh = 0, pinned = 0;
  int32_t n = 0, split = 0;
  uint64_t hh = 0;
  if (!rd(fl) || fl > 65536 || fl > file_size - raw.size())
    return false;
  std::string fpv(fl, '\0');
  if (fread(fpv.data(), 1, fl, f) != fl)
    return false;
  raw.insert(raw.end(), fpv.begin(), fpv.end());
  if (!rd(n) || n <= 0 || n > max_n)
    return false;
  if (!rd(ns) || ns > max_subs)
    return false;
  std::vector<int32_t> subs(ns);
  for (auto& x : subs)
    if (!rd(x))
      return false;
  if (!rd(ng) || ng != ngpu)
    return false;
  std::vector<uint64_t> sizes(ng);
  for (auto& x : sizes)
    if (!rd(x))
      return false;
  if (!rd(nh) || nh != ng * (1 + ns))
    return false;
  std::vector<uint64_t> hashes(nh);
  for (auto& x : hashes)
    if (!rd(x))
      return false;
  if (!rd(pinned) || !rd(split) || !rd(hh))
    return false;
  if ((size_t)n > (file_size - raw.size()) / sizeof(int32_t))
    return false;
  std::vector<int32_t> toks(n);
  for (auto& x : toks)
    if (!rd(x))
      return false;
  return decode(raw.data(), raw.size(), fingerprint, ngpu, max_subs, out);
}
} // namespace qkva
