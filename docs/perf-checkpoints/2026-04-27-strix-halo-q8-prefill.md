# 2026-04-27 Strix Halo Q8 Prefill

Hardware: Radeon 8060S / gfx1151, ROCm 7.2, toolbox `llama-rocm-7.2`.

Models:

- llama.cpp: `/home/kotdath/omp/personal/amd-strix-halo-toolboxes/models/Qwen3.5-9B-Q4_K_M.gguf`
- hipfire target: `~/.hipfire/models/qwen3.5-9b.mq4`
- hipfire draft: `~/.hipfire/models/qwen35-9b-dflash-mq4.hfq`

Binaries:

- `bench_qwen35_mq4`: `7e67241d0e31c09d810e89010f047d03`
- `dflash_spec_demo`: `f125f6d5669fe96cc2b8951313224ac6`
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

## DFlash Smoke

Prompt: `benchmarks/prompts/lru_cache_pep8_strict.txt` (`231` tokens), `--ctx 2048 --kv-mode q8 --no-adaptive-b --no-chatml`.

| Mode | Prefill tok/s | Decode tok/s | Notes |
| --- | ---: | ---: | --- |
| AR baseline | 329.0 | 48.00 | target-only greedy |
| DFlash | 327.3 | 101.45 | tau 8.75, accept rate 0.583 |

## Interpretation

Q8 KV does not explain the prefill gap: llama.cpp stays around 1.0k tok/s with Q8_0 KV, while hipfire stays GEMM-bound. The first actionable gfx1151-specific improvement is to avoid the current WMMA prefill GEMM path on gfx1150/gfx1151; dot2 is slower in theory but faster on this APU for the measured Qwen3.5 9B MQ4 prefill workload.
