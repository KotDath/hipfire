# 2026-04-27 Strix Halo Q8 Prefill

Hardware: Radeon 8060S / gfx1151, ROCm 7.2, toolbox `llama-rocm-7.2`.

Models:

- llama.cpp: `/home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf`
- hipfire target: `~/.hipfire/models/qwen3.5-9b.mq4`
- hipfire draft: `~/.hipfire/models/qwen35-9b-dflash-mq4.hfq`
- hipfire target converted from the Unsloth GGUF:
  `/home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/qwen3.5-9b-unsloth.hf4`

Binaries:

- `bench_qwen35_mq4`: `7cd9a6b059dc4d05b78ffb504470c680`
- `dflash_spec_demo`: `8df7bd8dc9d8f716a79fab8f7c95f825`
- `lru_cache_pep8_strict.txt`: `df5dedc8040ce70ba55080c4548e6024`

## llama.cpp Q8_0 KV

Command pattern:

```bash
llama-bench -m /home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf \
  -ngl 99 -p <N> -n 1 -r 3 -ctk q8_0 -ctv q8_0 -fa 1
```

`-fa 1` is required for `-ctv q8_0`; without it llama.cpp fails context creation.

| Prompt tokens | Prefill tok/s | Gen tok/s |
| ---: | ---: | ---: |
| 2048 | 1076.08 +/- 3.87 | 33.70 +/- 0.42 |
| 4096 | 1043.52 +/- 7.44 | 33.66 +/- 0.42 |

## hipfire Q8 KV

Command pattern:

```bash
HIPFIRE_KV_MODE=q8 ./target/release/examples/bench_qwen35_mq4 \
  ~/.hipfire/models/qwen3.5-9b.mq4 --prefill <N> --prefill-runs 3 --warmup 0 --gen 1
```

Before the gfx1151 routing change, q8 pp2048 measured `233.0 tok/s` median.

After routing gfx1150/gfx1151 away from WMMA prefill GEMM and onto dot2:

| Prompt tokens | Run tok/s | Median tok/s | Gen tok/s |
| ---: | --- | ---: | ---: |
| 2048 | 322.1, 320.5, 323.5 | 322.1 | 42.4 |
| 4096 | 307.1, 306.3, 302.5 | 306.3 | 38.0 |

After additionally selecting the existing residual `k2x32` WMMA variant on
gfx1150/gfx1151 only:

| Prompt tokens | Variant | Run tok/s | Median tok/s | Gen tok/s |
| ---: | --- | --- | ---: | ---: |
| 2048 | `k2` override | 340.6, 343.0, 337.2, 340.5, 336.7 | 340.5 | 38.0 |
| 2048 | auto `k2x32` | 360.1, 357.4, 357.4, 356.4, 355.7 | 357.4 | 39.1 |

This is a small but repeatable +5% over the same-binary `k2` baseline.

Re-check after the negative experiments were reverted:

| Prompt tokens | Run tok/s | Median tok/s | Gen tok/s |
| ---: | --- | ---: | ---: |
| 2048 | 355.8, 354.2, 354.0 | 354.2 | 42.8 |

Changing `HIPFIRE_PREFILL_MAX_BATCH` did not materially move pp2048:

| Max batch | Prefill tok/s |
| ---: | ---: |
| 64 | 361.5 |
| 128 | 356.3 |
| 192 | 358.9 |
| 256 | 357.2 |
| 384 | 359.0 |
| 512 | 351.6 |
| 1024 | 304.5 |
| 2048 | 306.8 |

The larger 1024/2048-token chunks regress instead of converging toward
llama.cpp, so the gap is not caused by hipfire's default chunk size being too
small.

## Model Artifact Check

The Unsloth GGUF was converted into hipfire `hf4` format and benchmarked with
the same Q8 KV prefill command. The result was `357.8 tok/s` median and `43.5`
gen tok/s, matching the native hipfire `.mq4` within noise. This rules out the
final Unsloth/GGUF model artifact as the cause of the 3x prefill gap; the gap
is in the hipfire execution format/kernels.

## Negative Experiments

These were tested and not kept:

- `HIPFIRE_ROCBLAS_ALL_ARCHS=1 HIPFIRE_ROCBLAS_MIN_BATCH=1`: `106.3 tok/s`
  prefill, much slower than the custom kernels on gfx1151.
- `BATCH_TILE=16` for dot2 qkv/qkvza/gate_up: regressed to about `223 tok/s`.
- `PREFILL_MAX_BATCH=512`: about `352 tok/s`, no useful gain over 256 and
  higher scratch footprint.
- `HIPFIRE_GRAPH=1`: about `348 tok/s`, so launch overhead is not the main
  limiter.
- `HIPFIRE_WMMA=1`: about `241 tok/s`; the old WMMA prefill path remains worse
  on gfx1151.
- 32x2 thread-block grouping for `gemm_gate_up_hfq4g256_dot2`: compiled and ran
  after fixing launch bounds, but measured about `354 tok/s`, i.e. no useful
  improvement over the simpler dot2 kernel.
- `HIPFIRE_WMMA=1` re-check after the residual routing change: about `225 tok/s`;
  gate/up WMMA alone was roughly 18 ms/call vs about 11 ms/call for dot2.
- Gate/up WMMA x32, modeled after the residual `k2x32` variant: about
  `235.5 tok/s`; still much slower than dot2.
- Gate/up dot2 with `BATCH_TILE=16` only: about `239 tok/s`; more batch rows per
  workgroup increased register pressure enough to lose badly.
- Gate/up dot2 multi-wave/LDS weight-sharing experiment: about `264.7 tok/s`;
  sharing did not compensate for occupancy and scheduling costs.
- `HIPFIRE_ROCBLAS_ALL_ARCHS=1 HIPFIRE_ROCBLAS_MIN_BATCH=1 ROCBLAS_USE_HIPBLASLT=1`:
  about `294 tok/s`; RDNA rocBLAS/hipBLASLt was still slower than the custom
  kernels for this format.
- Gate/up Q8-activation integer-dot experiment: compiled with `sudot4` on
  gfx1151 but measured about `309 tok/s`. Simple per-call Q8 activation
  quantization without MMQ-style row/batch tiling is a regression.
- Gate/up dot2 `BATCH_TILE=4`: the first apparent `~378 tok/s` result was an
  invalid host/kernel tile mismatch. The corrected implementation measured
  about `294 tok/s`, so it was reverted.
- Gate/up dot2 `__launch_bounds__(32, 16)`: measured `360.8 tok/s` median vs
  `359.7 tok/s` for the default `__launch_bounds__(32, 8)` in the same A/B
  session. This is measurement noise, not a useful tuning point.
- `HIPFIRE_MW16=1` residual path with per-call HFQ4->FP16 dequant: regressed to
  `~72-108 tok/s`. Dequantizing every prefill call is too expensive.
- Cached FP16-shadow residual MW16 prototype: reusing dequantized residual
  weights still measured only `~178 tok/s`, slower than the existing residual
  `k2x32` WMMA path on gfx1151.
- Skipping final prefill logits (temporary `HIPFIRE_PREFILL_SKIP_LOGITS=1`) only
  moved pp2048 to `~363 tok/s`. This matters for apples-to-apples methodology
  because llama.cpp's `llama-bench` calls `llama_batch_get_one(...)` with
  `logits=nullptr`, but it does not explain the gap.
- Gate/up Q8 activation prototype (`HIPFIRE_GATE_UP_Q8X=1`) quantized the
  activation matrix once per gate/up call and used gfx1151 integer dot
  instructions, but stayed in hipfire's row-wise work decomposition. It
  measured `~189.5 tok/s`, much slower than the dot2 baseline. This confirms
  that the llama.cpp-like win is not "Q8 activations" alone; it requires the
  MMQ row/batch tiling layout.
- Extending the existing `HIPFIRE_ROCBLAS_ALL_ARCHS=1` experiment so gate/up
  also used the FP16-shadow rocBLAS path measured `~348.2 tok/s` with
  `ROCBLAS_USE_HIPBLASLT=1`, still below the current dot2/k2x32 path. rocBLAS
  FP16 shadows are not a useful RDNA/Strix Halo bypass for this model.

## DFlash Smoke

Prompt: `benchmarks/prompts/lru_cache_pep8_strict.txt` (`231` tokens), `--ctx 2048 --kv-mode q8 --no-adaptive-b --no-chatml`.

| Mode | Prefill tok/s | Decode tok/s | Notes |
| --- | ---: | ---: | --- |
| AR baseline | 358.9 | 47.84 | target-only greedy |
| DFlash | 359.0 | 92.65 | tau 8.75, accept rate 0.583 |

## Interpretation

Q8 KV does not explain the prefill gap: llama.cpp stays around 1.0k tok/s with
Q8_0 KV, while hipfire stays GEMM-bound. Profiling after the gfx1151 routing
fix shows the remaining prefill time is dominated by 4-bit GEMM:

- `gemm_gate_up_hfq4g256_dot2`: ~49%
- `gemm_hfq4g256_residual_wmma_k2x32`: ~27%
- `gemm_qkvza_hfq4g256_dot2`: ~13%
- `gemm_qkv_hfq4g256_dot2`: ~3.5%

The gap also does not appear to be DPM/power-state driven. During the same
session, sysfs reported `power_dpm_force_performance_level=auto` and
`sclk=775Mhz`, but llama.cpp still measured `1069.93 tok/s` for pp2048. That
keeps the comparison valid: llama.cpp is fast in the same observed state where
hipfire is about `354-357 tok/s`.

The structural difference is the GEMM algorithm. llama.cpp's HIP/CUDA backend
routes quantized prompt processing through MMQ: it quantizes the activation
matrix into a `q8_1` layout and tiles both the batch dimension and the weight
rows. Hipfire's current hot kernels are still row-wise HFQ4 GEMMs: one output
row by an 8-token batch tile per workgroup, with per-row dequantization. Local
tweaks to this design did not close the gap.

Two details from llama.cpp matter for a faithful port:

- `tools/llama-bench/llama-bench.cpp::test_prompt` uses
  `llama_batch_get_one(tokens.data(), n_tokens)`.
- `src/llama-batch.cpp::llama_batch_get_one` sets `logits = nullptr`, so the
  prompt-processing number is not dominated by final logits. Hipfire's
  temporary no-logits test confirmed this is not the main gap.

The MMQ path then routes through `ggml_cuda_mul_mat_q`, first quantizing the
activation matrix with `quantize_mmq_q8_1_cuda`, then launching
`mul_mat_q_case<GGML_TYPE_Q4_K>`. For AMD, llama.cpp uses `mmq_y=128`,
`MMQ_ITER_K=256`, and Q4_K/Q8_1 vector-dot helpers with
`VDR_Q4_K_Q8_1_MMQ=8`. That design reuses the activation tile across many
weight rows; hipfire currently reloads/reprocesses activation data per output
row tile.

The measured safe improvements are:

1. Avoid the current WMMA prefill qkv/qkvza/gate_up path on gfx1150/gfx1151 and
   route those kernels to dot2.
2. Use residual `k2x32` only on gfx1150/gfx1151, where it is faster than `k2`;
   keep gfx1100 and other AMD targets on the previous auto path because the
   source comment records `k2x32` as slower on gfx1100.

The remaining gap to llama.cpp likely requires an MMQ-style 4-bit matmul
rewrite for the fused qkv/qkvza/gate_up GEMMs, not KV-cache tuning or larger
prefill chunks.
