# Qwen3.8-Flash-Next on two DGX Sparks

### Three checkpoints: `INT4-AutoRound`, the default and fastest — and `hibrid48` / `hibrid48-uncensored`. Switch with one line in `recipe.yaml`

[`INT4-AutoRound`](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) by
[@azampatti](https://github.com/azampatti) (default, every v5.2 number below): 5 of 512 experts per token, AutoRound int4
experts, fp8 side layers. [`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) and
[`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (the full 10-expert body,
NVFP4; the uncensored one is gated, no guardrails): richer, more creative answers at a little less speed. To switch: in
`recipe.yaml` comment the active `model:` line and uncomment another, then `./run.sh` — the kit applies each checkpoint's
settings itself. Details in [Which checkpoint](#which-checkpoint).

Two boxes, one model, RDMA. **v5.2: 93 tok/s average and 177 peak on a thinking-on request, 1,008 tok/s at 64 streams, 3,939
tok/s prefill at 128k, 0.36 s to the first token.** Three commands.

**Side by side with every other stack:** [myllmbox.com](https://myllmbox.com)

## v5.2 (2026-10-06)

v4.1 stays available: `git checkout v4.1`. The version now matches the solo kit's (same stack generation).

**New**

1. **Default checkpoint: [INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)** by
   [@azampatti](https://github.com/azampatti), as published, experts split whole across the two boxes (expert parallel). Its
   fp8 side layers load through his `vllm_fp8_hybrid` module (MIT, after [@Saren-Arterius](https://github.com/Saren-Arterius)'s
   [spark-dflash-hybrid-fp8](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)); its n-gram table is held on the
   GPUs, half per box, in the same 4-bit format as hibrid48's.
2. **The solo kit's stack, on two boxes:** RecoverSSM ([vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863)
   by [@jschmied](https://github.com/jschmied), ported to 0.30) and dynamic draft depth up to 7 — now inside full CUDA graphs,
   which two boxes need (the per-step cross-box exchange stays in the graph).
3. **One-shot RoCE all-reduce** for the small per-step exchanges: RoCEnante from
   [b12x](https://github.com/local-inference-lab/b12x) (Apache-2.0), wired in as in eugr's vLLM; larger messages stay on NCCL.
   **Both halves of each ConnectX-7 port** carry NCCL traffic. Both suggested and measured by
   [@sethforprivacy](https://github.com/sethforprivacy) (#5).
4. **GB10 skinny-GEMM plans** for the small bf16 layers — [@sethforprivacy](https://github.com/sethforprivacy)'s TP=2 table (#4),
   on by default in this image.
5. Upstream vLLM fixes backported: fused hyper-connection down projection + SiLU
   ([#58957](https://github.com/vllm-project/vllm/pull/58957)), the hyper-connection up projection kept on the skinny path
   ([#60027](https://github.com/vllm-project/vllm/pull/60027)), the QSA profiling KV released
   ([#58961](https://github.com/vllm-project/vllm/pull/58961)), the QSA logits workspace
   ([#57105](https://github.com/vllm-project/vllm/pull/57105) by Thien Tran).

**Measured** (2× DGX Spark, RDMA, image v5.2, the shipped `recipe.yaml`, INT4-AutoRound)

| | v5.2 |
|---|---|
| thinking-on request (pasture), c=1 | **93.2** tok/s average · **176.8** peak |
| peak at c=1 / 2 / 4 / 8 / 16 / 32 / 64 | **177 / 252 / 357 / 506 / 684 / 885 / 1,174** |
| average at c=1 / 2 / 4 / 8 / 16 / 32 / 64 (thinking off, mixed) | 100 / 155 / 244 / 335 / 484 / 718 / 968 |
| prefill, 128k-token prompt | **3,939** tok/s |
| first token, 1k prompt | **0.36** s |
| KV pool | **1,829,182** tokens |

**Per prompt** (thinking off, aggregate tok/s of all streams, averages of 3 runs; c=1 = one full answer, c≥2 = 300 s with every
stream kept busy; **peak** = the best 10-s window measured at that concurrency)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 | c=24 | c=32 | c=48 | c=64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **peak** | **176.8** | **252.1** | **357.0** | **430.6** | **506.1** | **600.8** | **684.3** | **791.9** | **885.3** | **1,021.4** | **1,174.0** |
| mixed (code + explanation) | 100.4 | 155.3 | 244.2 | 289.2 | 335.2 | 413.2 | 483.9 | 603.5 | 718.0 | 881.1 | 968.4 |
| structured output (JSON schema) | 124.4 | 208.8 | 311.9 | 377.5 | 442.6 | 538.8 | 614.5 | 703.6 | 783.3 | 915.8 | 977.1 |
| long prose (7,000-word story) | 74.0 | 122.1 | 195.2 | 242.0 | 288.0 | 352.5 | 413.0 | 490.3 | 555.0 | 638.2 | 699.5 |
| code (TypeScript) | 133.6 | 210.5 | 318.2 | 384.4 | 456.0 | 546.4 | 620.4 | 736.5 | 808.7 | 937.8 | 1,007.5 |

## Quality (measured on this model, thinking on)

All three checkpoints, lm-evaluation-harness against a running serve, **thinking on**, temperature 0.6 / top-p 0.95 / top-k
20, a 32k-token budget per answer, HumanEval complete and a fixed 200-question subset (seed 123123123) of the others — the same
763 questions for every column. Qwen publishes no numbers for these four tests (its card reports LiveCodeBench v6 91.9, GPQA
Diamond 91.7, IFBench 81.3, SWE-bench Pro 62.5).

| task | questions | INT4-AutoRound (v5.2) | hibrid48 | hibrid48-uncensored |
|---|---|---|---|---|
| HumanEval pass@1 | 164 | **93.3** | **95.7** | **94.5** |
| GSM8K exact match | 200 | **99.0** | **98.0** | **97.5** |
| IFEval prompt-level strict / instruction-level strict | 200 | **91.5** / 94.0 | **91.5** / 93.4 | **94.5** / 96.2 |
| MMLU-Pro (14 subjects, sampled by size) | 200 | **82.9** | **84.9** | **82.9** |
| answers that ran into the 32k budget while thinking (count as wrong) | 763 | 6 | 11 | 6 |

Subsets of 200 carry about ±3 points of sampling noise; published leaderboard numbers use other prompts, few-shot counts and
full sets — a sanity band, not a column.

## What changed

**v5.2 (2026-10-06): the INT4-AutoRound checkpoint and the solo stack on two boxes.** See [v5.2](#v52-2026-10-06).
v4.1 stays available: `git checkout v4.1`.

**v4.1 (2026-09-27): FlashInfer GDN prefill.** `gdn-prefill-backend` back to 0.30's default: **+5 % prefill from 8k to
256k tokens** (3,300 vs 3,140 tok/s at 32k), the decode ladder unchanged at every rung from 1 to 64 streams. Measured as
two complete runs of the same kit, one per backend. Thanks to sethforprivacy for the finding.

**v4 (2026-09-24): vLLM 0.30, five draft tokens, a 2.45M-token pool and 64 seats.** The engine moved to upstream vLLM
0.30, and speculative decoding was retuned on it:

1. **Five draft tokens, sampled.** The drafter now proposes 5 tokens instead of 4 and *samples* them from its own
   distribution (`draft_sample_method: probabilistic`) instead of always taking its top guess; the model checks them with
   the exact probability-ratio test and block verification. Output quality cannot change — the text follows the model's
   own distribution either way — but more drafts survive: 5.1 accepted per step on code (v3: 4.2 of 5). +15 % single-stream,
   +16 % at 32 streams. Measured one knob at a time: the sampled draft is the gain; block verification is neutral on top.
2. **NCCL on four channels** (`NCCL_MAX_NCHANNELS=4`). NCCL builds 64 channels on this pair of boxes and splits every large
   message across all of them; the GB10 has no GPUDirect RDMA, so each piece is copied through host memory. Four channels
   made the per-layer all-reduce 3× faster at 32–64 streams: +10 % at 32 streams, +7 % at 64, nothing at one.
3. **Half the n-gram table per box.** On 0.30 splitting it across the two boxes costs nothing measurable (it did on 0.29),
   and the 13.4G it frees per box goes to the KV pool: 28G → 41G per box, 1.71M → **2,450,356 pooled tokens**, 9.35× the full
   262K window. 64 seats instead of 32.
4. **Boots in about four minutes instead of about seventeen:** 0.30 serves this model without torch.compile, so the
   per-boot model compilation is gone.

The image also carries the fused multi-step draft for this model's attention (proposed upstream as vllm-project/vllm#58449;
+1–3 % engine steps at 1–12 streams), and the kit reads each box's RoCE GID index at launch — the index moves after a
reboot or link flap, and a stale one fails NCCL with "unhandled system error". fp8 KV is not in the v4 image. v3 stays
available: `git checkout v3` (image `…-cluster-vllm:v5`, vLLM 0.29, full table per box, 28G pin, 32 seats, fp8 KV opt-in).

**v3 (2026-09-13): vLLM 0.29 and a 4-bit output head.** Two changes, one number: the serve moved from the vendor's SM121 vLLM
pin to upstream vLLM 0.29 (the model is upstream now), and the checkpoint's 1.18 GiB bf16 output head — read ~5.4 times per
decode step by speculative decoding, 27 % of the step — became 0.33 GiB of NVFP4 (`hibrid48`). Engine steps **17.7 → 22.0**
single-stream; same body, same table, same drafter. v2's single-stream **peak** was 80 tok/s — on v3, 80 is the **average** of a
whole thinking-on request, reasoning included (79.5 over 38k tokens), and the code phase peaks at 107. KV is bf16 by default (1.71M pooled tokens on
the 28G pin): fp8 KV still works (2.85M) but cost 0.3 accepted tokens per step on this stack, so it is one commented line
away, not the default. Quality, measured with lm-evaluation-harness against this serve with thinking on: HumanEval 95.7,
GSM8K 98.0, IFEval 91.5, MMLU-Pro 84.9. v2.1 stays available: `git checkout v2.1` (image `…-cluster-vllm:v4`, hibrid47, fp8 KV).

**v2.1 (2026-09-08): fp8 KV.** Same model, same speed, 1.66× the KV pool: **2.85M pooled tokens** on the same 28G pin
(bf16 held 1.71M), so every seat carries more context and one request can run the full 262K window 10.9 times over.
Upstream vLLM's fp8-KV port for this model's QSA attention (PR #54846) is image patch 04; the engine still steps at
17.3/s single-stream (74 tok/s writing code, 54 thinking, 39,487 tokens in 11 minutes in one request). v2 stays
available: `git checkout v2` (image `…-cluster-vllm:v3`, bf16 KV).

**v2 (2026-09-06)** serves [myllmbox/Qwen3.8-Flash-Next-hibrid47](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid47):
the hibrid46 body with its 95 GB n-gram (PLE) table re-quantized to NVFP4 and held **resident on the GPU** — no CPU
offload worker, no per-step detour — split tensor-parallel across **two NVIDIA DGX Sparks (GB10, 119G unified memory
each)** over their ConnectX link, NCCL on RDMA. Against v1 (the int3 table in a CPU worker, same boxes, same tests):
**+7–11 % engine steps on every concurrency** and a table that draws 26 of 32 boss scenes where int3 drew half.
v1 stays available: `git checkout v1` in this repo (image `…-cluster-vllm:v2`, checkpoint hibrid46).

## Quick start

```bash
git clone https://github.com/myllmbox/qwen38-flash-next-cluster-recipe.git
cd qwen38-flash-next-cluster-recipe
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~99G, syncs it, serves on :8000
```

**Hugging Face token.** Anonymous downloads are rate-limited, and gated models (license-agreement repos, e.g. uncensored variants) refuse anonymous access. `run.sh` looks for `HF_TOKEN`, then `~/.cache/huggingface/token` (`hf auth login`), and asks for one when the repo is gated — after you accepted its agreement on the model page. Nothing is stored by the kit.

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks
with docker + the NVIDIA container runtime, connected by their ConnectX ports (a direct cable or a switch),
ssh from the head to the worker (a password once — `setup.sh` installs a key). With the weights on both boxes, a boot
reaches healthy in about four minutes (weights ~97 s, profile + KV + graph capture ~67 s).

**What `run.sh` does the first time:** no `cluster.env` yet → it runs [`setup.sh`](setup.sh), which asks for the
worker's ssh address, probes both boxes (interfaces, RDMA devices, GPU, docker), **discovers the interconnect**
(the interface pair that actually reaches the other box, tried by bound pings — never the management LAN),
checks the firewall **without root** (a throwaway listener on one box, one connect from the other, over the
interconnect — if it passes, nothing to open and no password is ever asked; if a box blocks its peer, it shows the
single `ufw allow from <peer>` it would run and asks first), creates the model/cache dirs on the worker, and writes
`cluster.env`. Rerun `./setup.sh` after re-cabling.

**All model configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights repo, port, KV budget, every
vLLM flag. The cluster flags (`--nnodes 2`, ranks, rendezvous address, TP=2, `--headless` on the worker) and the
per-box NCCL/gloo interface pins are added by `run.sh` from `cluster.env`; you never write them.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Which checkpoint

Three checkpoints run on this exact stack; `recipe.yaml` ships with the first active and the other two commented out under it.
Switching is swapping the `model:` line and the `kv-cache-memory` line under `vllm:`, then `./run.sh`; `run.sh` applies each checkpoint's settings itself:

| `model:` | what it is | on this kit |
|---|---|---|
| `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound` (default) | 5 of 512 experts per token, AutoRound int4 experts, fp8 side layers and n-gram table — every v5.2 number above | the fastest: 1,008 tok/s at 64 streams, 177 tok/s peak |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` | the full 10-expert body, calibrated, NVFP4 output head | ~10 % slower (below), richer and more creative answers |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` | OrcaRouter's abliterated (refusal-removed) body with the same head — **no guardrails**; research, red-teaming, private use behind your own moderation | same speed as hibrid48; quality table above |

hibrid48 runs ~10 % slower than the default: expect ~120 tok/s at 1 stream, ~590 at 16 and ~1,035 at 64. It is the lighter
checkpoint (~99 GB vs ~122 GB), so it leaves room to push the knobs: a bigger `kv-cache-memory`, more seats (see
[Tuning](#tuning-recipeyaml)).

The uncensored repo is **gated**: open its Hugging Face page, accept the agreement, then `hf auth login` (or `export
HF_TOKEN=…`) before `./run.sh` — the kit checks both and tells you what is missing. Running both checkpoints at different
times? Set `served-model-name` to something distinct (e.g. `Qwen/Qwen3.8-Flash-Next-Uncensored`) so clients and logs can tell
them apart. Both weigh ~99 GB; the first download of the second one is a full download (different body), the 8 table shards are
shared bytes. hibrid47 / hibrid47-uncensored (bf16 head) load on this image too, at v2.1 speed.

## The RDMA part (why the numbers are what they are)

NCCL will happily run over TCP sockets on the same ConnectX cable and never say so — the env vars look right, the
serve works, and every step is ~2× slower. A container needs three things to actually open the RDMA device:
`--device /dev/infiniband`, `--cap-add IPC_LOCK`, `--ulimit memlock=-1:-1`. `run.sh` passes them; `view.sh`
proves it by sampling the HCA's port counter against the interface's TCP byte counter while the model decodes
(RDMA moving, TCP flat = good). If `/dev/infiniband` is missing on a box (`rdma-core` not installed, or the link
is not a ConnectX one), the kit still runs, over TCP, and says so.

## Memory on a Spark: what the kit does about it

Unified memory means the GPU driver and the page cache share one pool, and the driver wants pages that are
**free**, not just reclaimable. After a few model loads the checkpoint's shards sit in the page cache and free
memory drops to ~1 GB while the next load allocates — the driver can stall on a copy that never completes, and the
boot looks hung at 100 % CPU. The kit never asks for your password; it stabilises memory with what a user may do:

- **waits** after removing old containers until both boxes report ≥ 100 GB available (unified memory takes
  30–60 s to come back after a container dies; launching earlier gives a phantom "CUDA out of memory"),
- **evicts its own checkpoint files from the page cache** before launch (`dd iflag=nocache`, no privileges),
- the weights load with `fastsafetensors` (`load-format` in `recipe.yaml`), which booted cleanly on every v4 launch; with
  vLLM's default loader instead, the image **drops each shard from the cache as soon as it has been consumed**, so the
  cache never balloons during the load itself.

Nothing to do on your side.

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0 on both boxes; shows the two commands, asks, then sudo prompts
```

`run.sh` checks the value on both boxes before every launch (reading needs no privilege) and prints a one-line
warning while it is not 0; it never applies it for you.

The kernel's background page compactor wakes on a low-free-memory box and migrates pages to build large
contiguous blocks. On a Spark the GPU's memory *is* those pages, so every migration first unmaps them from the
GPU: measured as a 4–5 s slowdown every ~37 s (the compactor's retry cycle), both GPUs idling together, no swap,
no clock change. A serving box allocates once at boot and gains nothing from the upkeep. The kit never runs
this for you (it needs root); it takes effect immediately, no restart.

## Tuning (recipe.yaml)

- **YaRN, up to 400k context** (supported): `recipe.yaml` carries a commented `max-model-len: 409600` + `hf-overrides` pair
  under `max-model-len` — swap it in. Config by [@vr8vr8](https://x.com/vr8vr8); verified by
  [@grau_marc](https://x.com/grau_marc): 100 % needle retrieval at 407,552 tokens, same speed.
- **`kv-cache-memory`** (bytes, per box): two lines in `recipe.yaml` — 29G for INT4-AutoRound (1,829,182 pooled tokens),
  33G for hibrid48 (2,079,675); swap them together with `model:`. Both leave ~11G available per box after graph capture —
  check `free -g` on both boxes after the first boot and back off if either shows swap in use. Do **not** take vLLM's
  "fully utilize" suggestion: unified memory over-commit has needed a power cycle.
- **`MBX_PLE_REPLICATE: "0"`** (env): half the n-gram table per box, exchanged per gather — same speed as the full table per
  box, and it is what pays for the KV pins above. `"1"` puts the full table on each box (~14G more per box); lower
  `kv-cache-memory` by the same amount.
- **`compilation-config`**: the model runner rounds every FULL-graph capture size up to a multiple of K+1 and drops the
  ones past the largest listed, so the list must reach `max-num-seqs × 8` = 512 (K=7). Shorten it only together with
  `max-num-seqs`; a list that stops short leaves the upper rungs decoding without CUDA graphs.
- **`block-size: 1632`**: required at K=5–7. This model's attention keeps a small ring of recent keys sized to hold the
  draft tokens (8 slots at K=4, 12 at K=5–7), and the ring must divide the attention block; vLLM's automatic block (1616)
  does not divide by 12 and the boot stops with "QSA ring capacity 12 must divide the attention block size 1616". Remove
  the line if you go back to K=4.
- **`NCCL_MAX_NCHANNELS: "4"`** (env): see [What changed](#what-changed). More channels only add host copies here; 2 and 8
  measured the same as 4, 1 was slower at 32 streams.
- **`engram-config: {"cpu_offload": false}`**: vLLM 0.30 would otherwise keep the n-gram table in pinned host memory;
  this image serves it from the GPU, the layout every number here was measured on.
- **`gdn-prefill-backend: flashinfer`**: +5 % prefill over `triton`, same decode speed.
- **`load-format: fastsafetensors`**: the loader every v4 boot was measured with. Remove it for vLLM's default loader.
- **`gpu-memory-utilization`** 0.70: with the pin set it does not size the KV pool.
- **`max-num-seqs`**: 64 — ~38k tokens of pool per seat, 12 tok/s per stream, 793 tok/s aggregate. Each running request
  also pins part of the pool the moment it is admitted, regardless of length: the GDN recurrent state, 36 layers ×
  (2 + K) blocks, held for rollback of rejected draft tokens (7 blocks at K=5). At 64 streams of this test the pool filled
  to 99 % after about three minutes; at 48 it peaked at 81 %. Set 48 if your load is many long answers at once; a smaller
  K = more seats (2 + K).
- **fp8 KV** is not in the v4 image (the patch has not been re-ported to 0.30 — upstream PR #54846 is still open);
  `git checkout v3` for it.
- **`speculative-config`** K=5, `draft_sample_method: probabilistic`, `rejection_sample_method: block`: acceptance ~5.1
  on code, ~3.9 over a whole thinking-on request, cap 6.0. Measured one change at a time against K=5 with the defaults:
  the sampled draft +7 % single-stream (thinking off), +1.5 % thinking on, +9 % at 32 streams; block verification neutral
  alone and on top. K=4 (`{"method":"mtp","num_speculative_tokens":4}`, capture list to 320, no `block-size`) is the
  previous setting. A bf16 drafter was A/B'd and rejected (acceptance +0.01, −3 % steps, +3.4G).
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits realistic multi-image requests.
- **`host: 127.0.0.1`**: the cluster runs on host networking, so the API would otherwise be on every interface.
  Set `0.0.0.0` to expose it on the LAN.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.
- **`patches`** (server): optional vLLM patches from [`patches/`](patches/), off by default — e.g. `patches: hermes-chat`
  for the Hermes agent (contributed by [@yume-arasaki](https://github.com/yume-arasaki)). Applied at launch over the image's files; the image itself is unchanged.

## Optional exact GDN replay build

[Exact native GDN replay](exact-gdn-replay/README.md) provides a pinned derived-image
build for this two-GB10 configuration. It reduces speculative decode state-copy
traffic while preserving native FP32 round points and existing cache capacity.
The package includes differential correctness tests, explicit scope and memory
costs, measurement conditions, and instructions for preserving a rollback. The
standard image and recipe defaults remain unchanged.

## What's in the image

`myllmbox/qwen38-flash-next-cluster-vllm:v5.2` — the solo kit's v5.2 image (vLLM 0.30.0 with our patches: the n-gram table
as a GPU parameter, the 4-bit output head, RecoverSSM, dynamic draft depth, INT4-AutoRound support) plus the two-box layer:

1. **dynamic draft depth in full CUDA graphs** — every depth 3–7 captured, no concurrency cap.
2. **INT4-AutoRound n-gram table** converted fp8 → NVFP4 at load, resident and split across the two boxes.
3. **one-shot RoCE all-reduce** — [b12x](https://github.com/local-inference-lab/b12x)'s RoCE collectives (Apache-2.0) for
   messages up to 2 MB, NCCL above.
4. **GB10 skinny-GEMM plans** for TP=2 (by [@sethforprivacy](https://github.com/sethforprivacy)) and the upstream fused
   hyper-connection kernels (vllm-project/vllm#58957, #60027, #58961).

Digest: `sha256:697d4364853312f3b50354020eb99cb12585228c9d16d1288261edb9631fa93b`

## The full box

This kit serves one model across two boxes, plain. The same model also runs under [myllmbox](https://myllmbox.com) with a
public HTTPS tunnel, dashboard, keepalive and multi-model management — same image, same weights.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products must display
the model name; Model-as-a-Service businesses need a separate Qwen license). Kit scripts and image patches: MIT.
