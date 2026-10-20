#!/usr/bin/env python3
"""GPU equivalence test for the small-batch decode patch.

Run inside the patched image with the checkout mounted:

  docker run --rm --gpus all -v "$PWD:/work" "$IMAGE" \
    /work/bench/test_small_batch_decode.py

The test checks both lossless substitutions in ``qwen3_5-small-batch-decode``:

* the row-split bf16 SiLU*up kernel is bit-identical to vLLM's stock CUDA op
  for query batches in its 1..64-token fast path, and both fp16 and batches
  above 64 still delegate to vLLM's stock op;
* a speculative GDN recurrence fed strided q/k/v column views is bit-identical
  to the same recurrence fed contiguous copies, including its updated state.

Exit 0 is a pass or a designed non-CUDA skip, and 1 is a failed check. This is
a GPU test, so it is listed in bench/README.md but not run by CPU-only CI.
"""

from __future__ import annotations

import sys

import torch


def _fail_exact(name: str, got: torch.Tensor, want: torch.Tensor) -> bool:
    if torch.equal(got, want):
        print(f"  PASS  {name}")
        return True
    mismatch = (got != want).sum().item()
    max_abs = (got.float() - want.float()).abs().max().item()
    print(f"  FAIL  {name}: {mismatch} differing elements, max abs {max_abs:g}")
    return False


def check_silu() -> int:
    from vllm.model_executor.layers.qwen27_small_ops import SMALL_TOKENS

    failures = 0
    for dtype in (torch.bfloat16, torch.float16):
        for tokens in (1, 2, 3, 8, 32, SMALL_TOKENS, SMALL_TOKENS + 1):
            for width in (128, 1024, 17_408):
                x = torch.randn(tokens, width * 2, device="cuda", dtype=dtype) * 4
                stock = torch.empty(tokens, width, device="cuda", dtype=dtype)
                candidate = torch.empty_like(stock)
                torch.ops._C.silu_and_mul(stock, x)
                torch.ops.vllm.qwen27_silu_and_mul(candidate, x)
                failures += not _fail_exact(
                    f"silu {dtype} tokens={tokens} width={width}", candidate, stock
                )
    return failures


def check_gdn_strides() -> int:
    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    torch.manual_seed(7301)
    dtype = torch.bfloat16
    tokens, heads, value_heads = 4, 2, 4
    key_dim = value_dim = 128
    state_slots = tokens + 1

    packed = torch.randn(
        tokens,
        2 * heads * key_dim + value_heads * value_dim,
        device="cuda",
        dtype=dtype,
    )
    q_strided = packed[:, : heads * key_dim].unflatten(-1, (heads, key_dim)).unsqueeze(0)
    k_strided = (
        packed[:, heads * key_dim : 2 * heads * key_dim]
        .unflatten(-1, (heads, key_dim))
        .unsqueeze(0)
    )
    v_strided = (
        packed[:, 2 * heads * key_dim :]
        .unflatten(-1, (value_heads, value_dim))
        .unsqueeze(0)
    )
    assert q_strided.stride(1) == packed.stride(0)
    assert k_strided.stride(1) == packed.stride(0)
    assert v_strided.stride(1) == packed.stride(0)

    def recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        state = torch.zeros(
            state_slots, value_heads, value_dim, key_dim, device="cuda", dtype=dtype
        )
        state[0].normal_(mean=0.0, std=0.05)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=torch.randn(value_heads, device="cuda", dtype=torch.float32),
            a=torch.randn(tokens, value_heads, device="cuda", dtype=dtype),
            b=torch.randn(tokens, value_heads, device="cuda", dtype=dtype),
            dt_bias=torch.randn(value_heads, device="cuda", dtype=torch.float32),
            q=q,
            k=k,
            v=v,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
            ssm_state_indices=torch.arange(
                1, tokens + 1, device="cuda", dtype=torch.int32
            ).unsqueeze(0),
            num_accepted_tokens=torch.ones(1, device="cuda", dtype=torch.int32),
            use_qk_l2norm_in_kernel=True,
        )

    # The helper draws internal tensors, so make both arms use the same values
    # by seeding each call and passing identical cloned initial states.  The
    # random A_log/a/b/dt_bias streams are deterministic under that seed.
    torch.manual_seed(991)
    strided_out, strided_state = recurrence(q_strided, k_strided, v_strided)
    torch.manual_seed(991)
    contiguous_out, contiguous_state = recurrence(
        q_strided.contiguous(), k_strided.contiguous(), v_strided.contiguous()
    )
    failures = int(
        not _fail_exact(
            "GDN output: strided views vs contiguous copies",
            strided_out,
            contiguous_out,
        )
    )
    failures += int(
        not _fail_exact(
            "GDN final state: strided views vs contiguous copies",
            strided_state,
            contiguous_state,
        )
    )
    return failures


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP(by design): CUDA is unavailable")
        return 0
    try:
        import vllm.model_executor.layers.qwen27_small_ops  # noqa: F401
    except ImportError as error:
        print(f"SKIP(by design): qwen3_5-small-batch-decode is not applied ({error})")
        return 0

    print("== small-batch decode equivalence")
    failures = check_silu()
    failures += check_gdn_strides()
    print(f"small-batch decode test: {'OK' if not failures else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
