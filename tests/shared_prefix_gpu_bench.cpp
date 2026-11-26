// Explicit, bounded host-registration and H2D DMA bench for shared PREFIX
// pages. Run only after local_capacity.py --brief says B's queue is zero.
// Caller chooses the device (t71: GPU 7), runs `nice -n 19 ionice -c3`, and
// stops if B logs an error. Device allocation is capped at 128 MiB; total wall
// time at 55 s.
#include <cuda_runtime.h>

#include <cerrno>
#include <fcntl.h>
#include <linux/magic.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

static double now_s() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static double quantile(std::vector<double> v, double p) {
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  const size_t i = std::min(
      v.size() - 1, (size_t)std::max(0.0, std::ceil(p * v.size()) - 1.0));
  return v[i];
}
static void report(const char *mode, const std::vector<double> &reg_ms,
                   const std::vector<double> &copy_ms, size_t bytes) {
  const double mib = bytes / (1024.0 * 1024.0);
  printf("GPU_BENCH mode=%s register_ms n=%zu p50=%.3f p95=%.3f\n", mode,
         reg_ms.size(), quantile(reg_ms, 0.50), quantile(reg_ms, 0.95));
  printf("GPU_BENCH mode=%s h2d_gb_s n=%zu p50=%.3f p95=%.3f\n", mode,
         copy_ms.size(),
         copy_ms.empty() ? 0.0
                         : mib / (quantile(copy_ms, 0.50) / 1000.0) / 1024.0,
         copy_ms.empty() ? 0.0
                         : mib / (quantile(copy_ms, 0.95) / 1000.0) / 1024.0);
}
static bool smaps_pages(void *base, size_t bytes,
                        unsigned long long &shmem_huge_kb,
                        unsigned long long &file_huge_kb) {
  shmem_huge_kb = file_huge_kb = 0;
  FILE *f = fopen("/proc/self/smaps", "r");
  if (!f)
    return false;
  char *line = nullptr;
  size_t cap = 0;
  bool in = false;
  const uintptr_t lo = (uintptr_t)base, hi = lo + bytes;
  while (getline(&line, &cap, f) > 0) {
    unsigned long long a = 0, b = 0;
    if (sscanf(line, "%llx-%llx", &a, &b) == 2) {
      in = ((uintptr_t)a == lo && (uintptr_t)b >= hi);
      continue;
    }
    if (!in)
      continue;
    if (sscanf(line, "ShmemPmdMapped: %llu kB", &shmem_huge_kb) == 1)
      continue;
    if (sscanf(line, "FilePmdMapped: %llu kB", &file_huge_kb) == 1)
      continue;
  }
  free(line);
  fclose(f);
  return true;
}
struct Mapping {
  int fd = -1;
  void *p = MAP_FAILED;
  size_t n = 0;
  std::string path;
  ~Mapping() {
    if (p != MAP_FAILED)
      munmap(p, n);
    if (fd >= 0)
      close(fd);
    if (!path.empty())
      unlink(path.c_str());
  }
};
// QE_BENCH_RW: 0 (default) read-only mapping; 1 writable mapping; 2 writable
// for registration, then mprotect(PROT_READ) while registered.
static int rw_mode() {
  const char *v = getenv("QE_BENCH_RW");
  return v && (*v == '1' || *v == '2') ? *v - '0' : 0;
}
static bool make_mapping(const std::string &dir, const char *mode, int advice,
                         size_t bytes, Mapping &m) {
  std::string pattern = dir + "/.qpin-gpu-bench-XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  m.fd = mkstemp(name.data());
  if (m.fd < 0)
    return false;
  m.path = name.data();
  if (ftruncate(m.fd, (off_t)bytes) != 0)
    return false;
  m.n = bytes;
  m.p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, m.fd, 0);
  if (m.p == MAP_FAILED || madvise(m.p, bytes, advice) != 0)
    return false;
  // Fault pages only after the advice has been applied, so each mode has a
  // fresh backing file and a fair 2M/4K trial.
  memset(m.p, 0x5a, bytes);
  if (rw_mode() == 0 && mprotect(m.p, bytes, PROT_READ) != 0)
    return false;
  (void)mode;
  return true;
}
static void run_mode(const char *mode, int advice, const std::string &dir,
                     int device, void *dev, size_t bytes, unsigned flags,
                     double wall_start) {
  Mapping m;
  if (!make_mapping(dir, mode, advice, bytes, m)) {
    printf("GPU_BENCH mode=%s skipped: tmpfs mapping/advice failed: %s\n", mode,
           strerror(errno));
    return;
  }
  unsigned long long shmem_huge_kb = 0, file_huge_kb = 0;
  smaps_pages(m.p, bytes, shmem_huge_kb, file_huge_kb);
  printf("GPU_BENCH mode=%s pages shmem_pmd_kb=%llu file_pmd_kb=%llu\n", mode,
         shmem_huge_kb, file_huge_kb);
  cudaStream_t stream = nullptr;
  cudaError_t ce = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (ce != cudaSuccess) {
    printf("GPU_BENCH mode=%s skipped: stream: %s\n", mode,
           cudaGetErrorString(ce));
    return;
  }
  std::vector<double> reg_ms, copy_ms;
  for (int i = 0; i < 7 && now_s() - wall_start < 55.0; i++) {
    double t = now_s();
    ce = cudaHostRegister(m.p, bytes, flags);
    if (ce != cudaSuccess && (flags & cudaHostRegisterReadOnly)) {
      (void)cudaGetLastError();
      ce = cudaHostRegister(m.p, bytes, cudaHostRegisterPortable);
      if (ce == cudaSuccess)
        printf("GPU_BENCH mode=%s registration_fallback=portable\n", mode);
    }
    reg_ms.push_back((now_s() - t) * 1000.0);
    if (ce != cudaSuccess) {
      printf("GPU_BENCH stop mode=%s device=%d register: %s\n", mode, device,
             cudaGetErrorString(ce));
      break;
    }
    // QE_BENCH_RW=2: register writable, then drop CPU write access; the pin
    // must survive and DMA must still read the pages.
    if (rw_mode() == 2 && mprotect(m.p, bytes, PROT_READ) != 0) {
      printf("GPU_BENCH stop mode=%s mprotect after register: %s\n", mode,
             strerror(errno));
      (void)cudaHostUnregister(m.p);
      break;
    }
    ce = cudaMemcpyAsync(dev, m.p, bytes, cudaMemcpyHostToDevice, stream);
    if (ce == cudaSuccess)
      ce = cudaStreamSynchronize(stream);
    if (ce != cudaSuccess) {
      printf("GPU_BENCH stop mode=%s warm H2D: %s\n", mode,
             cudaGetErrorString(ce));
      (void)cudaHostUnregister(m.p);
      break;
    }
    for (int j = 0; j < 11 && now_s() - wall_start < 55.0; j++) {
      t = now_s();
      ce = cudaMemcpyAsync(dev, m.p, bytes, cudaMemcpyHostToDevice, stream);
      if (ce == cudaSuccess)
        ce = cudaStreamSynchronize(stream);
      copy_ms.push_back((now_s() - t) * 1000.0);
      if (ce != cudaSuccess)
        break;
    }
    const cudaError_t ue = cudaHostUnregister(m.p);
    if (ce != cudaSuccess || ue != cudaSuccess) {
      printf("GPU_BENCH stop mode=%s H2D=%s unregister=%s\n", mode,
             cudaGetErrorString(ce), cudaGetErrorString(ue));
      break;
    }
  }
  report(mode, reg_ms, copy_ms, bytes);
  cudaStreamDestroy(stream);
}
int main(int argc, char **argv) {
  const double wall_start = now_s();
  const int device = argc > 1 ? atoi(argv[1]) : 0;
  int count = 0;
  cudaError_t ce = cudaGetDeviceCount(&count);
  if (ce != cudaSuccess || device < 0 || device >= count) {
    printf("GPU_BENCH skipped: requested device %d unavailable (count=%d, "
           "cuda=%s)\n",
           device, count, cudaGetErrorString(ce));
    return 77;
  }
  ce = cudaSetDevice(device);
  if (ce != cudaSuccess) {
    printf("GPU_BENCH skipped: cudaSetDevice(%d): %s\n", device,
           cudaGetErrorString(ce));
    return 77;
  }
  constexpr size_t bytes = 128u << 20; // <= 256 MB device-memory cap
  void *dev = nullptr;
  ce = cudaMalloc(&dev, bytes);
  if (ce != cudaSuccess) {
    printf("GPU_BENCH skipped: cudaMalloc 128 MiB: %s\n",
           cudaGetErrorString(ce));
    return 77;
  }
  const char *shared = getenv("QE_SHARED_L2_DIR");
  const std::string dir = shared && *shared ? shared : "/dev/shm";
  struct statfs fs{};
  if (statfs(dir.c_str(), &fs) != 0 ||
      (unsigned long)fs.f_type != (unsigned long)TMPFS_MAGIC) {
    printf("GPU_BENCH skipped: shared directory is not a mounted tmpfs (%s)\n",
           dir.c_str());
    cudaFree(dev);
    return 77;
  }
  int readonly_supported = 0;
  const cudaError_t attr = cudaDeviceGetAttribute(
      &readonly_supported, cudaDevAttrHostRegisterReadOnlySupported, device);
  const unsigned flags =
      cudaHostRegisterPortable |
      (attr == cudaSuccess && readonly_supported ? cudaHostRegisterReadOnly
                                                 : 0u);
  printf("GPU_BENCH device=%d count=%d bytes=%zu readonly_supported=%d attr=%s "
         "tmpfs=%s max_device_bytes=%zu wall_cap_s=55\n",
         device, count, bytes, readonly_supported, cudaGetErrorString(attr),
         dir.c_str(), bytes);
  run_mode("thp", MADV_HUGEPAGE, dir, device, dev, bytes, flags, wall_start);
  if (now_s() - wall_start < 55.0)
    run_mode("4k", MADV_NOHUGEPAGE, dir, device, dev, bytes, flags, wall_start);
  printf("GPU_BENCH wall_s=%.2f (owner ceiling 60s)\n", now_s() - wall_start);
  cudaFree(dev);
  return 0;
}
