# RTX 3090 Ti, MTP k=2: small-batch decode kernels

This is a paired A/B of upstream HyperQwen plus two launch reductions from
[`TheRealBluesun/HyperQwen-3090ti`](https://github.com/TheRealBluesun/HyperQwen-3090ti),
ported to the vLLM 0.31 tree. It is not a change to speculation depth, KV dtype,
sampling, or model weights.

## Configuration

- RTX 3090 Ti, 24 GB, sm86, WSL2, 350 W power limit
- Swift 1.5 Qwen3.8 27B W4A16 AutoRound derivative, vision tower on
- 96,506-row Han+CJK MTP draft head, `SPEC=mtp DRAFT_TOKENS=2`
- `CTX=long PTH_FA=1`, `int8_per_token_head` KV, `EMBED_UVA=1`
- `MAX_LEN=240000 MAX_SEQS=4 MAX_NUM_BATCHED_TOKENS=2048 GPU_UTIL=0.93`
- Prefix cache and tools on; 242,345-token KV pool

The two changes are:

- `qwen3_5-small-batch-decode.patch`: a row-split bf16 Triton SiLU*up kernel for
  up to 64 query tokens, and strided q/k/v views in the speculative GDN
  recurrence instead of reshape copies and a concatenation. They correspond to
  Bluesun patches 09 (`1199731`) and 10 (`a332c5e`) and keep the arithmetic and
  output dtype unchanged.

## Decode result

The same server, prompts, sampling and model were used for both arms. Each cell
is the median of five rounds at 256 requested output tokens. Wall speed includes
each request's prompt processing; engine speed is the server's own decode-time
counter.

| active requests | baseline wall tok/s | patched wall tok/s | change | baseline engine tok/s | patched engine tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 84.72 | 86.31 | +1.9% | 96.82 | 96.26 |
| 2 | 155.25 | 164.73 | +6.1% | 100.21 | 103.76 |
| 4 | 217.50 | 247.13 | +13.6% | 69.76 | 74.78 |

The C1 engine rate is within run-to-run noise; concurrent decode is the clear
gain. Acceptance varied with the generated text across the five rounds, so a
separate 15-sample pass was used for the arithmetic check: both arms measured
54.87% acceptance and 2.10 accepted tokens per verification step. The engine
median was 94.66 tok/s before and 105.58 tok/s with the patch.

## Verification

`python bench/test_small_batch_decode.py` checks, on CUDA:

- the bf16 SiLU*up fast path and fp16 fallback against vLLM's stock CUDA op at
  1, 2, 3, 8, 32, 64 and 65 query tokens, across three activation widths;
- the speculative GDN recurrence with strided q/k/v views against contiguous
  copies, comparing both output and the updated recurrent state bit for bit.

The patch also dry-applies with zero fuzz to a clean vLLM 0.31 tree with the
HyperQwen series applied.

## Not included

A PTH planner change that reduced empty decode-grid CTAs was measured and
rejected. It cut C2 engine throughput from about 100 to 70 tok/s and C4 from
about 70 to 60 tok/s in this profile, with a simultaneous acceptance drop. No
part of that experiment is in the patch series.
