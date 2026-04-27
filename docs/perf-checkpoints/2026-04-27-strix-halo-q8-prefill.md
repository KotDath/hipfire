# 2026-04-27 Strix Halo Q8 Prefill

Hardware: Radeon 8060S / gfx1151, ROCm 7.2, toolbox `llama-rocm-7.2`.

Models:

- llama.cpp: `/home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf`
- hipfire target: `~/.hipfire/models/qwen3.5-9b.mq4`
- hipfire draft: `~/.hipfire/models/qwen35-9b-dflash-mq4.hfq`

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

The measured safe improvements are:

1. Avoid the current WMMA prefill qkv/qkvza/gate_up path on gfx1150/gfx1151 and
   route those kernels to dot2.
2. Use residual `k2x32` only on gfx1150/gfx1151, where it is faster than `k2`;
   keep gfx1100 and other AMD targets on the previous auto path because the
   source comment records `k2x32` as slower on gfx1100.

The remaining gap to llama.cpp likely requires an MMQ-style 4-bit matmul
rewrite for the fused qkv/qkvza/gate_up GEMMs, not KV-cache tuning or larger
prefill chunks.
