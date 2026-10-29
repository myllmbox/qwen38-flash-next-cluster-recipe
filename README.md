# Qwen3.8-Flash-Next on two DGX Sparks

### v5.4: NVIDIA's [FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) checkpoint on vLLM 0.32 — 169 tok/s peak on one thinking-on request, 409,600-token context, 2.16M-token KV pool

Two boxes, one model, RDMA. The default checkpoint is
[nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local), served as shipped.
**v5.4: 84 tok/s average and 169 peak on a thinking-on request, 655–736 tok/s at 32 streams, 3,049 tok/s prefill at 128k,
0.35 s to the first token on a cached 128k prompt.** Three commands.

**Side by side with the one-box kit** — every number, gauntlet page and quality verdict: [myllmbox.com/?a=mbx-v54-tp2&b=mbx-v54](https://myllmbox.com/?a=mbx-v54-tp2&b=mbx-v54)

## v5.4 (2026-10-28)

Same checkpoint; `git checkout v5.3.1` for the previous one.

1. **Deeper drafts.** One request drafts up to 12 tokens; a ceiling by live requests keeps batches shallower
   (1 → 12, 2–3 → 9, 4–5 → 8, 6–11 → 7, 12–13 → 6, 14+ → 5). The depth policy is set in `recipe.yaml` env.
2. **bf16 KV cache and GDN state.** The deep drafts land about twice as often as with fp8 KV. Pool: 2,156,606 tokens
   (5.27 full 400k requests).
3. **The drafter scores the full vocabulary with an NVFP4 copy of the output head**, half per box. The model verifies with
   NVIDIA's bf16 head.
4. **32 seats** (v5.3: 64).

| | v5.4 | v5.3 |
|---|---|---|
| thinking-on request (pasture + fish), c=1: average / peak | **83.8** / **169.0** tok/s | 65.2 / 92.3 |
| code (TypeScript): c=1 / c=32 | **119.8** / **735.9** tok/s | 80.8 / 588.1 |
| mixed prompt (thinking off): c=1 / c=16 / c=32 | **78.8** / **438.0** / **655.0** tok/s | 65.6 / 376.6 / 560.1 |
| KV pool | 2,156,606 tokens (bf16) | 4,182,535 (fp8) |
| prefill, 128k-token prompt | 3,049 tok/s | 3,171 |
| first token: 1k prompt / 128k prompt again | 0.53 / **0.35** s | 0.44 / — |

**Per prompt** (thinking off, aggregate tok/s of all streams, averages of 3 runs; c=1 = one full answer, c≥2 = 300 s with every
stream kept busy; **peak** = the best 10-s window measured at that concurrency)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 | c=24 | c=32 |
|---|---|---|---|---|---|---|---|---|---|
| **peak** | **158.2** | **223.5** | **319.5** | **387.5** | **439.5** | **546.4** | **634.5** | **792.2** | **915.8** |
| mixed (code + explanation) | 78.8 | 123.5 | 194.4 | 242.9 | 291.1 | 364.9 | 438.0 | 559.9 | 655.0 |
| structured output (JSON schema) | 108.2 | 178.0 | 263.9 | 336.6 | 393.6 | 487.9 | 538.4 | 639.6 | 729.1 |
| long prose (7,000-word story) | 66.8 | 104.9 | 167.7 | 211.3 | 253.2 | 311.7 | 367.0 | 446.4 | 508.3 |
| code (TypeScript) | 119.8 | 192.5 | 283.7 | 352.3 | 401.0 | 484.6 | 547.7 | 648.9 | 735.9 |

## v5.3.1 (2026-10-23)

Same checkpoint and image base as v5.3; `git checkout v5.3` for the previous one.

1. **Agent turns resume from the cache.** Prefill checkpoints for the GDN layers ([vLLM #60050](https://github.com/vllm-project/vllm/pull/60050))
   and the n-gram short conv, cache hits every 64 tokens. A repeated 200k prompt recomputes 124 tokens (v5.3: ~12,500);
   the next turn of a conversation recomputes only the new turn.
2. **No quantization of our own.** The drafter runs as shipped (bf16); v5.3's FP8 drafter projections are gone.
   Draft acceptance 3.48 (v5.3: 3.30) on the thinking-on pasture.
3. **MXFP8 on b12x as NVIDIA ships it** (bf16 activations on decode, MXFP8 on prefill).

| | v5.3.1 | v5.3 |
|---|---|---|
| first token, 200k context already cached | 0.95 s | 4.1 s |
| recomputed on a repeated 200k prompt | 124 | ~12,500 |
| first token, 200k cold | 68.8 s | 63.4 s (197k) |
| thinking-on pasture, avg / peak tok/s | 66.4 / 91.9 | 66.8 / 92.3 |

## v5.3 (2026-10-21)

v5.2 stays available: `git checkout v5.2` (INT4-AutoRound on vLLM 0.30).

**New**

1. **vLLM 0.32.0**, the solo kit's v5.3 stack on two boxes: RecoverSSM
   ([vllm#58863](https://github.com/vllm-project/vllm/pull/58863) by [@jschmied](https://github.com/jschmied)), dynamic draft
   depth 1–7 in full CUDA graphs, the fused multi-step MTP draft.
2. **Default checkpoint: NVIDIA's FP4-Local**, weights as shipped: NVFP4 experts with compressed scales (they stay
   compressed on the GPU), MXFP8 dense layers. The n-gram table stays on the GPUs, half per box.
3. **Faster drafts**: FP8 drafter projections, a two-stage draft head and draft-depth bookkeeping kept on the GPU — by
   [@Alssndr0](https://github.com/Alssndr0). All exact: every draft is verified.
4. **400k context by default** (YaRN; config by [@vr8vr8](https://x.com/vr8vr8), verified by
   [@grau_marc](https://x.com/grau_marc)), with the M-RoPE cache fix from
   [vllm#62409](https://github.com/vllm-project/vllm/pull/62409) by [@wtogami](https://github.com/wtogami) built into the
   image. **fp8 KV cache**: 4,182,535 tokens in 36 GB per box.
5. Carried over from v5.2: the one-shot RoCE all-reduce ([b12x](https://github.com/local-inference-lab/b12x), Apache-2.0)
   and both halves of each ConnectX-7 port, suggested and measured by [@sethforprivacy](https://github.com/sethforprivacy)
   (#5, #6). His TP=2 skinny-GEMM table (#4) is in the image, off by default: on FP4-Local it measured the same as vLLM 0.32's
   own GB10 table.

**Measured** (2× DGX Spark, RDMA, image v5.3, the shipped `recipe.yaml`)

| | v5.3 (FP4-Local) | solo v5.3 (one box) |
|---|---|---|
| thinking-on request (pasture + fish), c=1: average / peak | **65.2** / **92.3** tok/s | 54.0 / 82.3 |
| gauntlet pages (pasture + fish) | **9 / 12** | 9 / 12 |
| mixed prompt (thinking off): c=1 / c=16 / c=64 | 65.6 / 376.6 / **755.3** tok/s | 54.2 / 266.3 / — |
| max context per request | **409,600** tokens | 409,600 |
| KV pool | **4,182,535** tokens (fp8) | 1,661,476 |
| prefill, 128k-token prompt | **3,171** tok/s | 2,301 |
| first token, 1k prompt | **0.44** s | 0.62 s |

**Per prompt** (thinking off, aggregate tok/s of all streams, averages of 3 runs; c=1 = one full answer, c≥2 = 300 s with every
stream kept busy; **peak** = the best 10-s window measured at that concurrency)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 | c=16 | c=24 | c=32 | c=48 | c=64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **peak** | **96.2** | **145.7** | **233.8** | **286.5** | **334.8** | **417.5** | **489.2** | **591.0** | **684.8** | **851.1** | **934.1** |
| mixed (code + explanation) | 65.6 | 108.4 | 159.8 | 201.9 | 241.6 | 320.4 | 376.6 | 482.6 | 560.1 | 660.6 | 755.3 |
| structured output (JSON schema) | 81.2 | 131.9 | 210.5 | 260.0 | 307.4 | 392.1 | 451.6 | 550.0 | 620.4 | 731.4 | 810.3 |
| long prose (7,000-word story) | 55.0 | 91.0 | 147.0 | 185.2 | 224.5 | 278.8 | 328.6 | 408.7 | 469.1 | 546.8 | 608.7 |
| code (TypeScript) | 80.8 | 126.3 | 194.9 | 244.6 | 291.1 | 368.0 | 425.4 | 520.8 | 588.1 | 699.7 | 793.3 |

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

## Quality (the v5.2 checkpoints, thinking on)

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

**v5.4 (2026-10-28): drafts up to 12 deep, bf16 KV, 32 seats.** See [v5.4](#v54-2026-10-28).
**v5.3.1 (2026-10-23): agent turns resume from the cache, no own quantization.** See [v5.3.1](#v531-2026-10-23).
**v5.3 (2026-10-21): NVIDIA's FP4-Local on vLLM 0.32, 400k context, fp8 KV.** See [v5.3](#v53-2026-10-21).
v5.2 stays available: `git checkout v5.2`.

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
./run.sh        # first run: sets the cluster up (asks for the 2nd box), downloads ~102G, syncs it, serves on :8000
```

**Hugging Face token.** Anonymous downloads are rate-limited, and gated models (license-agreement repos, e.g. uncensored variants) refuse anonymous access. `run.sh` looks for `HF_TOKEN`, then `~/.cache/huggingface/token` (`hf auth login`), and asks for one when the repo is gated — after you accepted its agreement on the model page. Nothing is stored by the kit.

`./stop.sh` stops both boxes. `./view.sh` shows live stats plus the RDMA proof. Requirements: two DGX Sparks
with docker + the NVIDIA container runtime, connected by their ConnectX ports (a direct cable or a switch),
ssh from the head to the worker (a password once — `setup.sh` installs a key). With the weights on both boxes, a boot
reaches healthy in about six minutes; the first one compiles kernels into `cache/` and takes about fifteen.

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

Every v5.4 number above is NVIDIA's [FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) (~102 GB),
the default in `recipe.yaml`. [INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound)
by [@azampatti](https://github.com/azampatti) — the v5.2 default, 5 of 512 experts per token — is one commented line below it;
the image carries its support, but it has not been measured on v5.4 (it needs a smaller `kv-cache-memory`: ~122 GB of weights).
[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) and
[`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (gated, no guardrails) run on the
same image by swapping the `model:` line; not measured on v5.4 either.
For the v5.2 numbers: `git checkout v5.2`.

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
- the weights load with the image's own loader (`load-format: mbx`), which **drops each shard from the cache as soon as it
  has been consumed**, so the cache never balloons during the load itself.

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

- **400k context** is the default (`max-model-len: 409600` + the YaRN `hf-overrides` line). For the model's native window:
  `max-model-len: 262144` and remove `hf-overrides`.
- **`kv-cache-memory`** (bytes, per box): 34G = 2,156,606 pooled bf16 tokens (5.27 full 400k requests); the head box keeps ~4 GB free under load. Check `free -g` on both boxes after the first boot and back off if either shows swap in use. Do **not** take
  vLLM's "fully utilize" suggestion: unified memory over-commit has needed a power cycle.
- **`max-num-seqs: 32`**: every one of the 32 seats ran at the same time through the whole test set. The capture list
  must reach the widest verify the depth ceiling allows: 32 × 6 = 192 (up to 5 drafts + 1 at 14+ streams).
- **`block-size: 3456`**, **`prefix-match-unit: 64`**: the block is a multiple of 16 (the attention's ring of recent keys at
  K=12) and of 64 (the prefill-checkpoint grid); cache hits land every 64 tokens. Leave them.
- **`mm-encoder-tp-mode: data`**: the vision tower whole on each box. Split across two, one of its NVFP4 layers lands on a
  width the quantization kernel refuses and the boot stops.
- **`MBX_PLE_REPLICATE: "0"`** (env): half the n-gram table per box. `"1"` puts the full table on each box (~14G more per box);
  lower `kv-cache-memory` by the same amount.
- **`MBX_SKINNY_GEMM_SM12X`** (env, unset = off): `"1"` swaps vLLM 0.32's own GB10 plans for the small bf16 layers for
  [@sethforprivacy](https://github.com/sethforprivacy)'s TP=2 table (#4). Same engine step rate on FP4-Local (20.7 vs 20.8 steps/s).
- **`NCCL_MAX_NCHANNELS: "4"`** (env): more channels only add host copies here; 2 and 8 measured the same as 4, 1 was slower
  at 32 streams.
- **`MBX_DEPTH_*`** (env): dynamic draft depth, chosen live from the acceptance of the deepest draft; `MBX_DEPTH_CEIL` caps it by
  live requests; `MBX_DEPTH_MODE: off` = always `num_speculative_tokens`.
- **`kv-cache-dtype: auto`** (bf16): `fp8` doubles the pool for less speed; with fp8 set `num_speculative_tokens` to 7.
- **`speculative-config`** K=12, `draft_sample_method: probabilistic`, `rejection_sample_method: block`: the text follows the
  model's own distribution either way; sampled drafts survive more often.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits realistic multi-image requests.
- **`host: 127.0.0.1`**: the cluster runs on host networking, so the API would otherwise be on every interface.
  Set `0.0.0.0` to expose it on the LAN.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.
- **`patches`** (server): optional vLLM patches from [`patches/`](patches/), off by default — e.g. `patches: hermes-chat`
  for the Hermes agent (contributed by [@yume-arasaki](https://github.com/yume-arasaki)). Applied at launch over the image's files; the image itself is unchanged.

## What's in the image

`myllmbox/qwen38-flash-next-cluster-vllm:v5.4` — the solo kit's v5.4 image (vLLM 0.32.0 with our patches: RecoverSSM, dynamic
draft depth in full graphs, the compressed-scale NVFP4 expert kernel, the n-gram table on the GPU, GDN + n-gram prefill checkpoints, the
M-RoPE YaRN fix, the NVFP4 drafter head copy) plus the two-box layer:

1. **one-shot RoCE all-reduce** — [b12x](https://github.com/local-inference-lab/b12x)'s RoCE collectives (Apache-2.0) for
   messages up to 2 MB, NCCL above.
2. **GB10 TP=2 skinny-GEMM plans** by [@sethforprivacy](https://github.com/sethforprivacy), opt-in.
3. **INT4-AutoRound n-gram table** converted fp8 → NVFP4 at load, resident and split across the two boxes.

Digest: `sha256:46cd09cdcd96ef69887df91d6e546cb43a02991ae2642a358930aec52418bfc8`

## The full box

This kit serves one model across two boxes, plain. The same model also runs under [myllmbox](https://myllmbox.com) with a
public HTTPS tunnel, dashboard, keepalive and multi-model management — same image, same weights.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products must display
the model name; Model-as-a-Service businesses need a separate Qwen license). Kit scripts and image patches: MIT.
