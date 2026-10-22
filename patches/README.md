# Optional vLLM patches

Off by default. Turn one on by listing its name in `recipe.yaml`:

```yaml
server:
  patches: hermes-chat          # space-separated, applied in this order
```

At launch `run.sh` copies each file a patch touches out of the image, applies the patch, copies the result to the
worker and mounts it read-only over the image's file on both boxes. The image is never changed; an empty
`patches` = the stock image. A patch that does not fit the image stops `run.sh` before anything starts.

| patch | what it does |
|---|---|
| `hermes-chat` | Hermes agent: reads its `{"reasoning": {…}}` object (thinking on/off, effort) and makes an omitted temperature greedy. Contributed by [@yume-arasaki](https://github.com/yume-arasaki) (#2) |
| `gb10-skinny-gemm` | GB10 (SM12x) plans for vLLM's Qwen4Exp skinny decode GEMM: decode-sized BF16 projections run the CuTe-DSL kernel (240-255 GB/s) instead of cuBLAS's SM80 WMMA fallback (125-225 GB/s). 1-stream decode step 47.6 → 46.1 ms on hibrid48 (with five vLLM backports in the same measurement); `MBX_SKINNY_GEMM_SM12X=0` turns it off. Contributed by [@sethforprivacy](https://github.com/sethforprivacy) |
| `roce-oneshot-ar` | TP=2 decode-size all-reduces (≤ 1 MiB) over RoCE with b12x's RoCEnante one-shot (~15 µs vs NCCL ring LL's 38-93 µs), in eager mode and CUDA graphs; bit-identical to NCCL's sum at TP=2. With dual-rail NCCL: 1-stream decode step −1.9 ms, 8 / 16 streams +6 / +5 % on hibrid48. Opt-in twice: the patch, then `ROCE_AR: "1"` and `PYTHONPATH: "/cache/b12x"` in `env:` with [b12x](https://github.com/local-inference-lab/b12x) @ `e4084d2e` cloned into `cache/b12x` on both boxes (details in the patch header). Contributed by [@sethforprivacy](https://github.com/sethforprivacy) |

## Adding one

A unified diff with paths relative to the `vllm` package (`--- a/entrypoints/…`, `+++ b/entrypoints/…`), made against
the image in `recipe.yaml`. Lines before the first `---` are a free-text description.
