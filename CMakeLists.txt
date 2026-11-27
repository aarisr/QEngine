cmake_minimum_required(VERSION 3.24)
# nvcc 13.1 rejects the system GCC 16; GCC 15 works
if(NOT CMAKE_CXX_COMPILER AND EXISTS /usr/bin/g++-15)
  set(CMAKE_CXX_COMPILER /usr/bin/g++-15)
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)
endif()
# GPU targets: when the user gave none (-DCMAKE_CUDA_ARCHITECTURES="86;120" or CUDAARCHS win), the GPUs of this
# machine (nvidia-smi compute capabilities, as CMake's "native"; 86 = RTX 3090 tuned, 89 = RTX 4090, 120 = RTX 5090),
# so a source build runs native code on every card instead of JIT-compiled compute_86 PTX on a newer one; without a
# visible GPU: 86. Set before project().
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
  set(_qe_archs "")
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                  OUTPUT_VARIABLE _qe_caps RESULT_VARIABLE _qe_rc ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_qe_rc EQUAL 0 AND _qe_caps)
    string(REPLACE "\n" ";" _qe_caps "${_qe_caps}")
    foreach(_c IN LISTS _qe_caps)
      string(STRIP "${_c}" _c)
      string(REPLACE "." "" _c "${_c}")
      if(_c MATCHES "^[0-9]+$")
        list(APPEND _qe_archs ${_c})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES _qe_archs)
    # sm_86 always in: a build on a 4090 / 5090 workstation then still runs on the 3090s it is deployed to (review L7)
    if(_qe_archs AND NOT "86" IN_LIST _qe_archs)
      list(APPEND _qe_archs 86)
    endif()
  endif()
  if(_qe_archs)
    message(STATUS "QEngine: CUDA architectures ${_qe_archs} (the GPUs of this machine; -DCMAKE_CUDA_ARCHITECTURES=... overrides)")
  else()
    set(_qe_archs 86)
    message(STATUS "QEngine: no GPU visible (nvidia-smi): CUDA architecture 86 (RTX 3090); for other cards set "
                   "-DCMAKE_CUDA_ARCHITECTURES, e.g. \"86;89;120\"")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES "${_qe_archs}" CACHE STRING "CUDA SASS targets (86 = RTX 3090, 89 = RTX 4090, 120 = RTX 5090)")
endif()
project(qengine LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=znver3")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -lineinfo --expt-relaxed-constexpr --extra-device-vectorization -Xptxas -warn-spills")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# build identity (remote diagnosis): the [box] start-up line of Flash-Next and the server's /api/stats `build` /
# start-up log (build/qe_build.txt next to the libraries, server/app.py version_info). Set at BUILD time, so a
# `cmake --build` after a commit or an edit names the code it compiled: git hash of the source tree (+ "-dirty" with
# uncommitted changes to tracked files). -DQE_GIT_HASH=... or the QE_GIT_HASH environment variable at configure time
# override it (trees without .git: release exports, the Docker build stage).
if(NOT DEFINED QE_GIT_HASH AND DEFINED ENV{QE_GIT_HASH})
  set(QE_GIT_HASH "$ENV{QE_GIT_HASH}")
endif()
file(WRITE ${CMAKE_BINARY_DIR}/qe_build_id.cmake [=[
set(id "${FORCED}")
if(id STREQUAL "" OR id STREQUAL "unknown")
  set(id "unknown")
  execute_process(COMMAND git -C "${SRC}" rev-parse --show-toplevel OUTPUT_VARIABLE top
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE rc)
  if(rc EQUAL 0 AND top)
    file(REAL_PATH "${top}" top)
    file(REAL_PATH "${SRC}" src)
    if(top STREQUAL src)  # the tree itself, not a repository it was unpacked into
      execute_process(COMMAND git -C "${SRC}" rev-parse --short=7 HEAD OUTPUT_VARIABLE h
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      execute_process(COMMAND git -C "${SRC}" status --porcelain -uno OUTPUT_VARIABLE d
                      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
      if(h)
        set(id "${h}")
        if(d)
          string(APPEND id "-dirty")
        endif()
      endif()
    endif()
  endif()
endif()
file(MAKE_DIRECTORY "${OUT}/gen")
file(WRITE "${OUT}/qe_build.txt.tmp" "${id}\n")
file(WRITE "${OUT}/qe_build_id.h.tmp" "#define QE_BUILD_ID \"${id}\"\n")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUT}/qe_build.txt.tmp" "${OUT}/qe_build.txt")
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${OUT}/qe_build_id.h.tmp" "${OUT}/gen/qe_build_id.h")
]=])
add_custom_target(qe_build_id ALL
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_SOURCE_DIR} -DOUT=${CMAKE_BINARY_DIR} "-DFORCED=${QE_GIT_HASH}"
          -P ${CMAKE_BINARY_DIR}/qe_build_id.cmake
  BYPRODUCTS ${CMAKE_BINARY_DIR}/qe_build.txt ${CMAKE_BINARY_DIR}/gen/qe_build_id.h
  COMMENT "build id" VERBATIM)
string(REPLACE ";" "," QE_ARCH_STR "${CMAKE_CUDA_ARCHITECTURES}")
add_compile_definitions(QE_CUDA_ARCHS="${QE_ARCH_STR}")
find_package(CUDAToolkit REQUIRED)
# std::thread (async shared-prefix publisher)
find_package(Threads REQUIRED)

file(GLOB KERNELS CONFIGURE_DEPENDS src/kernels/*.cu)
# Hash kernel source content and target architectures, excluding only the host-side tuning
# implementation/API. Thus tune-cache edits preserve the existing device-kernel fingerprint.
file(GLOB_RECURSE QE_KERNEL_SET_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cuh"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.inc"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.inc")
list(SORT QE_KERNEL_SET_FILES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${QE_KERNEL_SET_FILES})
set(QE_KERNEL_SET_MANIFEST "cuda_archs=${QE_ARCH_STR}\n")
function(qe_kernel_file_hash kernel_file kernel_rel output_var)
  if("${kernel_rel}" STREQUAL "src/kernels/gemm.cu")
    set(begin_marker "// GEMM_TUNE_CACHE_BEGIN")
    set(end_marker "// GEMM_TUNE_CACHE_END")
    set(strip_include "#include \"../gemm_tune_key.h\"\n")
  elseif("${kernel_rel}" STREQUAL "src/kernels/gemv.cuh")
    set(begin_marker "// GEMM_TUNE_API_BEGIN")
    set(end_marker "// GEMM_TUNE_API_END")
    set(strip_include "#include <vector>\n")
  elseif("${kernel_rel}" STREQUAL "src/gemm_tune_key.h" OR "${kernel_rel}" STREQUAL "src/archive_map.h" OR "${kernel_rel}" STREQUAL "src/archive_store_policy.h")
    set(kernel_hash "host-only-cache-helper")
    set(${output_var} "${kernel_hash}" PARENT_SCOPE)
    return()
  else()
    file(SHA256 "${kernel_file}" kernel_hash)
    set(${output_var} "${kernel_hash}" PARENT_SCOPE)
    return()
  endif()
  file(READ "${kernel_file}" kernel_source)
  string(FIND "${kernel_source}" "${begin_marker}" kernel_begin)
  string(FIND "${kernel_source}" "${end_marker}" kernel_end)
  if(kernel_begin LESS 0 OR kernel_end LESS kernel_begin)
    message(FATAL_ERROR "GEMM kernel-set markers missing from ${kernel_file}")
  endif()
  string(LENGTH "${end_marker}" end_length)
  math(EXPR suffix_start "${kernel_end} + ${end_length}")
  string(LENGTH "${kernel_source}" source_length)
  math(EXPR suffix_length "${source_length} - ${suffix_start}")
  string(SUBSTRING "${kernel_source}" 0 ${kernel_begin} kernel_prefix)
  string(SUBSTRING "${kernel_source}" ${suffix_start} ${suffix_length} kernel_suffix)
  set(kernel_source "${kernel_prefix}${kernel_suffix}")
  string(REPLACE "${strip_include}" "" kernel_source "${kernel_source}")
  string(SHA256 kernel_hash "${kernel_source}")
  set(${output_var} "${kernel_hash}" PARENT_SCOPE)
endfunction()
foreach(kernel_file IN LISTS QE_KERNEL_SET_FILES)
  file(RELATIVE_PATH kernel_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${kernel_file}")
  qe_kernel_file_hash("${kernel_file}" "${kernel_rel}" kernel_hash)
  string(APPEND QE_KERNEL_SET_MANIFEST "${kernel_rel}:${kernel_hash}\n")
endforeach()
string(SHA256 QE_GEMM_KERNEL_SET_HASH "${QE_KERNEL_SET_MANIFEST}")
# Only import v1 tables when the current device/kernel fingerprint is the known QEngine 1.6.2
# c844e301 baseline built for CUDA_ARCHS=86; later kernel changes reject the unhashed legacy file.
set(QE_GEMM_V1_LEGACY_KERNEL_SET_HASH "b8dbb40b1676621c23c98bf06e852a1a0e43920b823691e646710062bc05fa93")
add_compile_definitions(QE_GEMM_KERNEL_SET_HASH="${QE_GEMM_KERNEL_SET_HASH}"
                        QE_GEMM_V1_LEGACY_KERNEL_SET_HASH="${QE_GEMM_V1_LEGACY_KERNEL_SET_HASH}")
add_library(qengine SHARED ${KERNELS} src/capi.cpp src/config.cpp src/safetensors.cpp src/model.cpp src/dflash.cpp src/engine.cpp src/build_id.cpp)
target_include_directories(qengine PUBLIC src)
target_include_directories(qengine PRIVATE ${CMAKE_BINARY_DIR}/gen)
target_link_libraries(qengine PUBLIC CUDA::cudart CUDA::cuda_driver)
add_dependencies(qengine qe_build_id)  # qe_build.txt next to the library is current when it is linked

# Qwen3.8-Flash-Next engine (preview)
# kernels + safetensors without the 27B engine / C API (libqflash must not export libqengine's qe_* symbols)
add_library(qkernels OBJECT ${KERNELS} src/safetensors.cpp)
target_include_directories(qkernels PUBLIC src)
add_executable(qe_w8q src/tools/qe_w8q.cu src/build_id.cpp $<TARGET_OBJECTS:qkernels>)
add_dependencies(qe_w8q qe_build_id)
target_include_directories(qe_w8q PRIVATE ${CMAKE_BINARY_DIR}/gen)  # INT8 g128 of a bf16 matrix with the engine kernel (convert_flash_next.py --int8)
target_include_directories(qe_w8q PRIVATE src)
target_link_libraries(qe_w8q PRIVATE CUDA::cudart CUDA::cuda_driver)
add_executable(flash_run src/flash/flash.cu src/flash/flash_seq.cu src/flash/flash_pp.cpp src/flash/card_probe.cu src/flash/flash_main.cpp src/build_id.cpp $<TARGET_OBJECTS:qkernels>)
target_include_directories(flash_run PRIVATE src)
target_link_libraries(flash_run PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(flash_run PRIVATE ${CMAKE_BINARY_DIR}/gen)
add_dependencies(flash_run qe_build_id)
# C API of the Flash-Next engine for the server (server/qe_capi.h tier 0)
add_library(qflash SHARED src/flash/flash.cu src/flash/flash_seq.cu src/flash/flash_pp.cpp src/flash/card_probe.cu src/flash/flash_capi.cpp src/build_id.cpp $<TARGET_OBJECTS:qkernels>)
target_include_directories(qflash PRIVATE src)
target_link_libraries(qflash PRIVATE CUDA::cudart CUDA::cuda_driver Threads::Threads)
target_include_directories(qflash PRIVATE ${CMAKE_BINARY_DIR}/gen)
add_dependencies(qflash qe_build_id)

# CPU unit tests of the comm rules and the exchange flag protocol (no GPU): ctest
enable_testing()
foreach(t comm_rule_unit drv_gate_unit xchg_seq_unit sample_rng_unit archive_order_unit pool_order_unit kv_disk_unit disk_eager_unit qfn_pp_plan_unit load_drop_unit shared_prefix_unit df_adapt_unit mtp_share_unit host_pressure_unit slot_state_window_unit live_prefix_unit live_prefix_layout_unit live_prefix_driver_unit gemm_tune_key_unit idle_poll_unit archive_map_unit archive_store_policy_unit)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cpp)  # the Docker build stage has no tests/
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE src)
    add_test(NAME ${t} COMMAND ${t})
    if(t STREQUAL load_drop_unit)
      target_sources(${t} PRIVATE src/safetensors.cpp)
    endif()
    if(t STREQUAL kv_disk_unit OR t STREQUAL drv_gate_unit OR t STREQUAL host_pressure_unit OR t STREQUAL shared_prefix_unit OR t STREQUAL live_prefix_unit OR t STREQUAL live_prefix_driver_unit)
      find_package(Threads REQUIRED)
      target_link_libraries(${t} PRIVATE Threads::Threads)
    endif()
  endif()
endforeach()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/kv_arith_unit.cpp)
  add_executable(kv_arith_unit tests/kv_arith_unit.cpp)
  target_include_directories(kv_arith_unit PRIVATE src)
  add_test(NAME kv_arith_unit COMMAND kv_arith_unit)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/shared_prefix_gpu_bench.cpp)  # the Docker build stage has no tests/
  add_executable(shared_prefix_gpu_bench tests/shared_prefix_gpu_bench.cpp)
  target_link_libraries(shared_prefix_gpu_bench PRIVATE CUDA::cudart)
endif()

# GLM-5.3-Flash engine; only where src/glm is present
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/glm/glm.cu)
  find_package(CUDAToolkit REQUIRED COMPONENTS cublas)
  # [glm-rel-isa] portable CPU lane: no -march anywhere (plain x86-64); the lane kernels (glm_cpu_isa.cpp) are built
  # twice, AVX-512 (the Zen 4 set) and AVX2 (x86-64-v3), picked at lane start from cpuid (GLM_CPU_ISA=avx2 forces AVX2);
  # -mtune never changes the ISA
  foreach(isa avx512 avx2)
    add_library(glm_cpu_${isa} OBJECT src/glm/glm_cpu_isa.cpp)
    target_include_directories(glm_cpu_${isa} PRIVATE src)
    target_link_libraries(glm_cpu_${isa} PRIVATE CUDA::cudart)
    set_target_properties(glm_cpu_${isa} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_options(glm_cpu_${isa} PRIVATE -O3 -mtune=znver4)
  endforeach()
  target_compile_definitions(glm_cpu_avx512 PRIVATE GLM_ISA_AVX512=1)
  target_compile_definitions(glm_cpu_avx2 PRIVATE GLM_ISA_AVX512=0)
  add_library(qglm SHARED src/glm/glm.cu src/glm/glm_kernels.cu src/glm/glm_step.cu src/glm/glm_ar.cu src/glm/glm_moe8.cu src/glm/glm_gemm8.cu src/glm/glm_cpu.cpp $<TARGET_OBJECTS:glm_cpu_avx512> $<TARGET_OBJECTS:glm_cpu_avx2> src/glm/glm_plan.cpp src/glm/glm_capi.cpp src/glm/glm_ckpt.cu src/build_id.cpp $<TARGET_OBJECTS:qkernels>)
  target_include_directories(qglm PRIVATE src ${CMAKE_BINARY_DIR}/gen)
  add_dependencies(qglm qe_build_id)
  target_compile_definitions(qglm PRIVATE GLM_SRC_DIR="${CMAKE_SOURCE_DIR}")
  target_link_libraries(qglm PRIVATE CUDA::cudart CUDA::cuda_driver CUDA::cublas)
  set_source_files_properties(src/glm/glm_cpu.cpp PROPERTIES COMPILE_OPTIONS "-O3;-mtune=znver4")
endif()
