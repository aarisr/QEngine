#include "archive_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static int fails = 0;
#define EXPECT(x)                                                              \
  do {                                                                         \
    if (!(x)) {                                                                \
      printf("FAIL line %d: %s\n", __LINE__, #x);                              \
      ++fails;                                                                 \
    }                                                                          \
  } while (0)

// base64 -> bytes (test-only; the real on-disk header is embedded below)
static std::vector<char> b64_decode(const char* s) {
  auto val = [](char c) -> int {
    if (c >= 'A' && c <= 'Z')
      return c - 'A';
    if (c >= 'a' && c <= 'z')
      return c - 'a' + 26;
    if (c >= '0' && c <= '9')
      return c - '0' + 52;
    if (c == '+')
      return 62;
    if (c == '/')
      return 63;
    return -1;
  };
  std::vector<char> out;
  int acc = 0, bits = 0;
  for (const char* x = s; *x; ++x) {
    int v = val(*x);
    if (v < 0)
      continue;
    acc = (acc << 6) | v;
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back((char)((acc >> bits) & 0xff));
    }
  }
  return out;
}
// The 2060 header bytes of quarantine-0024-a/p_434a4a7a00cb2ca2_431_0.qkv (the real on-disk
// file after the 0024 crash: 176338956 bytes, n=431, ns=0, ng=3, pinned=1, split=0).
static const char kPinHdrB64[] =
      "UUtWQTAwMDT0AAAAdjN8L2FwcC9tb2RlbHMvUXdlbjMuOC0yN0ItSU5UNHxtdHA6L2FwcC9tb2RlbHMvUXdlbjMu"
      "OC0yN0ItSU5UNC9tb2RlbF9tdHBfZnQuc2FmZXRlbnNvcnN8MTg2MDMzODc2NTY6MTc5NTA4OTcyM3w4NDk0MDAz"
      "OTI6MTc5NTA4ODc4NnxnM3x8a3ZzcGxpdDEwMjQsNjcyLDY3MnxrdjR2MXw1NTg0MTc5Miw1NTg0NjkxMiw1NTgz"
      "NjY3Mnw1NTg0NjkxMiw1NTg1NzE1Miw1NTgzNjY3Mnw1NTg0MTc5Miw1NTg0NjkxMiw1NTgzNjY3Mq8BAAAAAAAA"
      "AwAAAACsdQMAAAAAAFiXAwAAAAAArHUDAAAAAAMAAAByAAT9qCV03rOJKnIPocXow+dQ51G5bN8BAAAAAAAAAB3J"
      "YfL5xGkj7cgDAOYhAADGAAAAFl8AAB4BAAAPEwAAcQEAAMwCAAA2AQAApgwAAA0AAADOMwAApgIAAIcbAAAyJQAA"
      "QQEAALMoAAALAAAAUBsAAI4WAAA2AQAAFwEAAEY/AAC1BwAALl8AAFZeAABqAQAADQAAAA8BAAD+IQAAxLAAABkA"
      "AAByBQAA0mgAAOtkAAC+AwAAiAcAAO4BAADlcgAAaBQAAJELAAANAAAAuM8AANQEAACtAgAAChMAAAsAAAC4BAAA"
      "CwAAAHgQAAALAAAA0gEAAO43AAANAAAADwEAANsJAAABAgAACAEAACOhAAChQwAApQEAAJzBAAD7EQAArioAAO4B"
      "AADlcgAAsEcAAA0AAAAPAQAAZR0AAMEUAAAZAAAAxgAAAAwAAAAxZQAAeAMAABcBAADlcgAAsEcAADYBAAD3DwAA"
      "FwEAAOAMAADGAAAADAAAAAIcAABbMgAALwEAAIYSAAC0AwAAFwEAAOURAAD3DwAAwhAAAJIDAABvTwAApSkAAMYA"
      "AAAMAAAAYjEAAJEUAACwRwAANgEAABgFAAAIAQAAUhEAALwZAADGAAAADAAAAKwKAACHKQAA1wEAAJ0BAADoAQAA"
      "VwIAAJEUAACIBwAACwAAAL4DAACoAQAANgEAADkLAAAXAQAA+gYAAB4MAAD3DwAAxgAAAAwAAADrZAAAvgMAAIgH"
      "AADuAQAAaBQAAJELAADXAQAAagMAABAjAADBGAAA0gEAAH2vAAAPAQAAqRsAAHYHAABIAQAAKAAAALMFAACSAwAA"
      "VwIAAIgHAAABAAAAnQEAABcBAADlcgAAGwMAAHEBAAC9IwAAsaoAADYBAAAXAQAA4AwAAA0AAAAPAQAABSkAANdR"
      "AAB6BgAAGQAAAJgUAABYFwAAmr4AAB6KAAAvAQAApgIAAPcPAAAZAAAAxgAAAAwAAAAhUQAAPgEAACYCAAALAAAA"
      "8D4AAAgAAABsAQAAuTUAAMYAAAAMAAAALo8AAD4BAAAKlQAA0gEAAFm4AAAIAAAAbAEAAHsmAAAKAwAAxgAAAAwA"
      "AACxrQAADgAAAO+BAABsAQAARV0AAMYAAAAMAAAAM58AALgBAABYFwAAykkAAD4BAAAnZwAAVSYAANwFAADuBQAA"
      "QQEAANYEAAAIAAAAxgAAAAwAAAAnFwAA8iUAADQFAAB1HwAAxgAAAAwAAADeLwAAxLAAABkAAAD0VwAAdAMAAFUm"
      "AABlEwAA7gUAAEEBAADWBAAAeAkAAIcUAAA+AQAAS3kAAAsAAACyBwAA8iUAAAsAAACEKwAACAAAAMYAAAAMAAAA"
      "qfMAABliAABZBwAAuTUAAA8BAAD+IQAAxLAAABkAAAAYIwAA62QAABcBAAD+BQAAP1cBAPcPAAANAAAA7gsAAM0P"
      "AAAOCwAAGQAAAMYAAAAMAAAA+jsAAHGsAACAAgAAcwMAAEQEAADoAQAAsAgAABQOAAAwDQAAKAAAAGQMAABACgAA"
      "37AAAEgBAAAkJgAA2AIAAHdZAAA0BAAAZwMAAMYAAAAMAAAALgUAAGMnAAAKAwAAOgEAAKYCAACKewAAGQcAAMYA"
      "AAAMAAAAoA4AAGNdAAA6AQAApgIAAMEUAADGAAAAuiUAAOQMAAAXAQAAAggAAPcPAAC4AQAAWBcAAJq+AAAeigAA"
      "DQAAAA8BAAD+IQAAxLAAABkAAAAlBAAAcQEAAAgBAADuTQAADAAAAOkVAACMEAEAgZUAAPkGAAANAAAAuM8AAKIJ"
      "AADEBgAA0BMAAPoRAAALAAAAYwsAADYBAABACgAA1wUAAAsAAADUDwAABWQAAAsAAADSAQAAKQMAALgBAACJDwAA"
      "EAQAAEgBAADLaQAA6AEAABAEAADYAgAANgEAAFRQAADSAQAASAEAACQmAADYAgAAVgUAAJ0BAAA0BAAAVgwAADEC"
      "AACEBAAAtQ8AAFclAAANAAAAcREAAPcPAABuBwAAfQEAAFIRAABBAQAAfgIAACJnAQANAAAADwEAACYCAABwgwAA"
      "xgAAAAwAAABQlQAALwEAABcBAADkOgEAmA8AAK4BAAAXAQAAhAQAAEMCAADgDAAAPgEAAEQAAADpBAAADQAAAAgB"
      "AADgDAAALwEAAAEgAAC/EwAACAEAAAEgAAD3DwAAGgAAANQoAAA9KwAA1CgAACkCAADGAAAADAAAAFgFAAAIAQAA"
      "5H8AAN4MAAD3bQAACAEAAK0HAACYDwAACwAAAMQGAAAXAQAA5H8AAGQHAACoAQAA3RIAAErIAAB4AwAAmwEAAC0G"
      "AAANAAAA7sgDAMYAAAA=";
static const char* kPinFp = "v3|/app/models/Qwen3.8-27B-INT4|mtp:/app/models/Qwen3.8-27B-INT4/model_mtp_ft.safete""nsors|18603387656:1795089723|849400392:1795088786|g3||kvsplit1024,672,672|kv4v1|5584""1792,55846912,55836672|55846912,55857152,55836672|55841792,55846912,55836672";
static const size_t kPinFull = 176338956ULL;  // the on-disk file's full size (header + payload)

int main() {
  qkva::Header pin;
  pin.fingerprint = "model-v4";
  pin.n = 4;
  pin.pinned = 1;
  pin.toks = {10, 11, 12, 13};
  pin.sizes = {4096, 2048};
  pin.hashes = {1, 2};
  EXPECT(qkva::valid(pin, 2, 4));
  auto bytes = qkva::encode(pin);
  qkva::Header decoded;
  EXPECT(qkva::decode(bytes.data(), bytes.size(), "model-v4", 2, 4, decoded));
  EXPECT(decoded.pinned == 1 && decoded.split == 0 && decoded.n == 4);
  EXPECT(decoded.toks == pin.toks && decoded.sizes == pin.sizes &&
         decoded.data_off == bytes.size());
  EXPECT(
      !qkva::decode(bytes.data(), bytes.size(), "stale-layout", 2, 4, decoded));
  EXPECT(
      !qkva::decode(bytes.data(), bytes.size() - 1, "model-v4", 2, 4, decoded));

  qkva::Header child;
  child.fingerprint = pin.fingerprint;
  child.n = 7;
  child.split = 4;
  child.toks = {10, 11, 12, 13, 20, 21, 22};
  child.sizes = {2048};
  child.hashes = {3};
  auto child_bytes = qkva::encode(child);
  qkva::Header child_decoded;
  EXPECT(qkva::decode(child_bytes.data(), child_bytes.size(), "model-v4", 1, 4,
                      child_decoded));
  EXPECT(child_decoded.split == 4 && child_decoded.pinned == 0);
  EXPECT(qkva::same_prefix(child_decoded, pin));
  qkva::Header bad_range = child;
  bad_range.split = bad_range.n;
  auto invalid_bytes = qkva::encode(bad_range);
  EXPECT(!qkva::decode(invalid_bytes.data(), invalid_bytes.size(), "model-v4",
                       1, 4, child_decoded));
  child_bytes[0] = 'X';
  EXPECT(!qkva::decode(child_bytes.data(), child_bytes.size(), "model-v4", 1, 4,
                       child_decoded));
  EXPECT(qkva::valid(child, 1, 4));
  qkva::Header suffix = child;
  suffix.pinned = 1;
  EXPECT(!qkva::valid(suffix, 1, 4));
  suffix = child;
  suffix.split = child.n;
  EXPECT(!qkva::valid(suffix, 1, 4));
  EXPECT(qkva::same_prefix(child, pin));
  qkva::Header absent = pin;
  absent.toks[2]++;
  EXPECT(!qkva::same_prefix(child, absent));
  absent.n = 3;
  absent.toks.resize(3);
  EXPECT(!qkva::same_prefix(child, absent));
  child.split = 8;
  EXPECT(!qkva::valid(child, 1, 4));
  EXPECT(qkva::ram_tier(true, false) == qkva::RamTier::none);
  EXPECT(qkva::ram_tier(true, true) == qkva::RamTier::prefix);
  EXPECT(qkva::ram_tier(false, true) == qkva::RamTier::ordinary);

  // 0024 on-disk regression: the header bytes of a real on-disk QKVA0004 pinned-prefix
  // file (27B archive cache, quarantined after the disk-cache start-up crash). The crash
  // was a null dereference right after this parse succeeded for a good file; the fix is
  // in Engine::disk_init (it now allocates the archive's disk state, like the QKVA0003
  // branch always did). This is the CPU side of the same contract: a good file loads,
  // a truncated or corrupt one is dropped - never a crash. decode_file is the parse
  // Engine::disk_init runs; the engine adds the payload-size check it can only do with
  // the worker models.
  std::vector<char> pinhdr = b64_decode(kPinHdrB64);
  EXPECT(pinhdr.size() == 2060);  // data_off of the on-disk file
  auto write_file = [](const std::string& path, const std::vector<char>& v) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
      return false;
    bool ok = v.empty() || fwrite(v.data(), 1, v.size(), f) == v.size();
    fclose(f);
    return ok;
  };
  char tmpl[] = "/tmp/qkva0024_XXXXXX";
  int tfd = mkstemp(tmpl);
  EXPECT(tfd >= 0);
  if (tfd >= 0)
    close(tfd);
  const std::string tf = tmpl;
  auto parse = [&](const std::vector<char>& bytes, const std::string& fprint,
                   size_t ngpu, size_t max_subs, int max_n, size_t file_size) {
    EXPECT(write_file(tf, bytes));
    FILE* f = fopen(tf.c_str(), "rb");
    EXPECT(f);
    if (!f)
      return qkva::Header{};
    qkva::Header m;
    EXPECT(qkva::decode_file(f, file_size, fprint, ngpu, max_subs, max_n, m));
    fclose(f);
    return m;
  };
  {
    // the good file: loads, every field of the real header verified
    qkva::Header m = parse(pinhdr, kPinFp, 3, 4, 262144, kPinFull);
    EXPECT(m.n == 431 && m.pinned == 1 && m.split == 0);
    EXPECT(m.toks.size() == 431 && m.sizes ==
           (std::vector<uint64_t>{58043392, 60250112, 58043392}));
    EXPECT(m.hashes.size() == 3 && m.data_off == 2060);
    EXPECT(m.fingerprint == std::string(kPinFp));
    EXPECT(qkva::valid(m, 3, 4));
  }
  // truncated anywhere in the header: refused (the write was cut by a power loss),
  // never a crash
  for (size_t cut : {size_t(1), size_t(8), size_t(9), size_t(100), size_t(355),
                     size_t(356), size_t(1000), size_t(2059)}) {
    std::vector<char> t(pinhdr.begin(), pinhdr.begin() + cut);
    qkva::Header m;
    EXPECT(write_file(tf, t));
    FILE* f = fopen(tf.c_str(), "rb");
    EXPECT(f);
    if (f) {
      EXPECT(!qkva::decode_file(f, t.size(), kPinFp, 3, 4, 262144, m));
      fclose(f);
    }
  }
  {
    // a flipped token id: the metadata checksum catches it
    std::vector<char> bad = pinhdr;
    bad[1000] ^= 1;
    qkva::Header m;
    EXPECT(write_file(tf, bad));
    FILE* f = fopen(tf.c_str(), "rb");
    EXPECT(f);
    if (f) {
      EXPECT(!qkva::decode_file(f, kPinFull, kPinFp, 3, 4, 262144, m));
      fclose(f);
    }
    // a flipped token id (another position): the checksum catches it too
    bad = pinhdr;
    bad[356] = (char)(bad[356] + 1);
    EXPECT(write_file(tf, bad));
    f = fopen(tf.c_str(), "rb");
    EXPECT(f);
    if (f) {
      EXPECT(!qkva::decode_file(f, kPinFull, kPinFp, 3, 4, 262144, m));
      fclose(f);
    }
    // a wrong magic: skipped like any foreign file
    bad = pinhdr;
    bad[0] = 'X';
    EXPECT(write_file(tf, bad));
    f = fopen(tf.c_str(), "rb");
    EXPECT(f);
    if (f) {
      EXPECT(!qkva::decode_file(f, kPinFull, kPinFp, 3, 4, 262144, m));
      fclose(f);
    }
    // the wrong engine (fingerprint), the wrong GPU count, a context below n: refused
    EXPECT(!qkva::decode_file(fopen(tf.c_str(), "rb"), kPinFull, "other-model", 3, 4,
                              262144, m));
    EXPECT(!qkva::decode_file(fopen(tf.c_str(), "rb"), kPinFull, kPinFp, 2, 4,
                              262144, m));
    EXPECT(!qkva::decode_file(fopen(tf.c_str(), "rb"), kPinFull, kPinFp, 3, 4,
                              100, m));
    // a file shorter than its own header: refused
    EXPECT(write_file(tf, pinhdr));
    EXPECT(!qkva::decode_file(fopen(tf.c_str(), "rb"), 1000, kPinFp, 3, 4, 262144, m));
    // an empty file: refused
    EXPECT(write_file(tf, {}));
    EXPECT(!qkva::decode_file(fopen(tf.c_str(), "rb"), 0, kPinFp, 3, 4, 262144, m));
  }
  // the split shape (synthesized with the same codec the writer uses): loads with split > 0
  {
    qkva::Header c;
    c.fingerprint = std::string(kPinFp);
    c.n = 7;
    c.split = 4;
    c.toks = {10, 11, 12, 13, 20, 21, 22};
    c.sizes = {2048};
    c.hashes = {3};
    auto cb = qkva::encode(c);
    qkva::Header m = parse(cb, kPinFp, 1, 4, 262144, cb.size());
    EXPECT(m.split == 4 && m.pinned == 0 && m.n == 7);
    EXPECT(qkva::valid(m, 1, 4));
  }
  std::string rm = "rm -f " + tf;
  (void)system(rm.c_str());
  printf(fails ? "archive_format_unit: %d FAILED\n"
               : "archive_format_unit: ok\n",
         fails);
  return fails ? 1 : 0;
}
