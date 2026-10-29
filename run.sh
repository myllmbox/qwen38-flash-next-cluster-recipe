#!/usr/bin/env bash
# Qwen3.8-Flash-Next (NVIDIA's FP4-Local by default, vLLM 0.32) on TWO DGX Sparks: TP=2 over the ConnectX link (RDMA).
# First run: no cluster.env → ./setup.sh (finds the second box, the interconnect, opens the firewall). Then:
# pull the image on both boxes, download the weights (once) and sync them to the worker, start the worker
# (--headless) and the head, wait healthy. Everything model-side is recipe.yaml; the boxes are cluster.env.
# OpenAI API on the head at :$PORT. ./stop.sh stops both, ./view.sh shows live stats + the RDMA proof.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source lib.sh
command -v docker >/dev/null || { echo "docker is required"; exit 1; }

# 0. boxes — none configured yet? set the cluster up first.
if ! have_cluster; then
  echo "· no cluster.env — running ./setup.sh first"
  ./setup.sh
fi
load_cluster
ssh_w true 2>/dev/null || { echo "✗ cannot reach the worker $WORKER — rerun ./setup.sh"; exit 1; }

IMAGE="$(rkey server image)";   PORT="$(rkey server port)";   HOST="$(rkey server host)";  HOST="${HOST:-127.0.0.1}"
HF_REPO="$(rkey server model)"; MPORT="$(rkey server master_port)"; MPORT="${MPORT:-25000}"
MODELS_DIR="$(rkey server models_dir)"; CACHE_DIR="$(rkey server cache_dir)"; CPUSET="$(rkey server cpuset)"
mkdir -p "$MODELS_DIR" "$CACHE_DIR"
MODELS_ABS="$(cd "$MODELS_DIR" && pwd)"; CACHE_ABS="$(cd "$CACHE_DIR" && pwd)"
LOCAL_NAME="$(basename "$HF_REPO")"; MODEL_DIR="$MODELS_ABS/$LOCAL_NAME"
ssh_w "mkdir -p '$MODELS_ABS' '$CACHE_ABS'"

# 1. image on BOTH boxes (a public tag — each box pulls; no-op when present)
echo "· image $IMAGE — head"; docker pull -q "$IMAGE" >/dev/null || docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "✗ cannot pull $IMAGE"; exit 1; }
echo "· image $IMAGE — worker"; ssh_w "docker pull -q '$IMAGE' >/dev/null || docker image inspect '$IMAGE' >/dev/null 2>&1" || { echo "✗ worker cannot pull $IMAGE"; exit 1; }

# 2. weights: ~102G, resumable — download on the head, then sync to the worker at the SAME path
# "complete" = the index is there AND no partial blob is left behind by an interrupted download (huggingface_hub keeps
# them under .cache/huggingface/download/*.incomplete and resumes them) — the index lands early, so it alone proves nothing.
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ] || [ -n "$(find "$MODEL_DIR/.cache" -name '*.incomplete' -print -quit 2>/dev/null)" ]; then
  hf_access "$HF_REPO" || exit 1
  echo "· downloading $HF_REPO -> $MODEL_DIR"
  if command -v hf >/dev/null; then
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  else
    TTY=""; [ -t 1 ] && TTY="-t"
    # the container runs as root — hand the files back to the host user afterwards (a root-owned .cache/ with 0600
    # files breaks the rsync to the worker; seen 2026-09-06)
    docker run --rm $TTY -e HF_TOKEN -v "$MODELS_ABS:/dl" --entrypoint python3 "$IMAGE" \
      -c "from huggingface_hub import snapshot_download; import subprocess; snapshot_download('$HF_REPO', local_dir='/dl/$LOCAL_NAME'); subprocess.run(['chown', '-R', '$(id -u):$(id -g)', '/dl/$LOCAL_NAME'], check=False)"
  fi
fi
if ! ssh_w "[ -f '$MODEL_DIR/model.safetensors.index.json' ]"; then
  echo "· syncing weights to the worker (one-time, ~102G over ssh — resumable, rerun if interrupted)"
  # .cache/ = huggingface_hub's download bookkeeping; the worker never reads it
  rsync -a --size-only --info=progress2 --exclude '.cache/' -e "ssh -o BatchMode=yes" "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
fi
echo "✓ weights on both boxes: $MODEL_DIR"
# The container mounts the kit's models/ folder at /models, nothing else — a symlink for models/<name> would dangle
# inside the container (its target is outside the mount) and vLLM would mistake the path for a Hugging Face repo id.
# So models/<name> must be a REAL directory: a download, or a hardlink copy (`cp -al /path/to/checkpoint models/<name>`,
# instant, zero extra space, same filesystem) — never a symlink.
for side in head worker; do
  if [ "$side" = head ]; then islink=$([ -L "$MODEL_DIR" ] && echo yes || echo no); else islink=$(ssh_w "[ -L '$MODEL_DIR' ] && echo yes || echo no"); fi
  if [ "$islink" = yes ]; then
    echo "✗ $side: $MODEL_DIR is a symlink — the container cannot follow it. Replace it with a real directory:"
    echo "    rm $MODEL_DIR && cp -al /path/to/$LOCAL_NAME $MODEL_DIR     # hardlink copy, instant"
    exit 1
  fi
done


# 4. compose the docker run for a rank. Host networking + the three RDMA flags (without them NCCL silently
#    falls back to TCP over the same cable — half the speed, no error). Per-box pins: NCCL/gloo on the
#    interconnect iface, VLLM_HOST_IP = that box's interconnect IP (the LAN must never carry cluster traffic).
ENVS=(); while IFS=$'\t' read -r k v; do [ -n "$k" ] && ENVS+=(-e "$k=$v"); done < <(rsection env)
FLAGS=(); while IFS=$'\t' read -r k v; do
  case "$v" in true) FLAGS+=("--$k");; false|null|"") ;; *) FLAGS+=("--$k" "$v");; esac
done < <(rsection vllm)

# 4a. INT4-AutoRound checkpoint (its fp8 n-gram table in ple-table/, its own MTP draft head): fp8 side layers + draft scale 2, the
#     table converted fp8 → NVFP4 at load (resident, half per box), its draft head (cache/draft-k10: links + small JSON, top-k 10,
#     the head's shared-expert width) on BOTH boxes, expert parallel (int4 group-128 experts cannot be TP-split for Marlin), no
#     global MoE backend (the draft head's bf16 experts refuse it)
if [ -d "$MODEL_DIR/ple-table" ]; then
  ENVS+=(-e VLLM_FP8_HYBRID=1 -e MBX_MTP_DRAFT_SCALE=2 -e "MBX_PLE_FP8_DIR=/models/$LOCAL_NAME/ple-table" -e MBX_PLE_FP8_TO_NVFP4=1)
  python3 - "$MODEL_DIR" "/models/$LOCAL_NAME" "$CACHE_ABS/draft-k10" <<'PY' || { echo "✗ could not build the draft folder"; exit 1; }
import json, os, shutil, struct, sys
snap, csnap, out = sys.argv[1:4]
wm = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
keep = {k: v for k, v in wm.items() if k.startswith(("mtp.", "lm_head.")) or k.endswith("embed_tokens.weight")}
if os.path.lexists(out):
    shutil.rmtree(out)
os.makedirs(out)
for f in sorted(os.listdir(snap)):
    if f in ("config.json", "model.safetensors.index.json", "ple-table", "fast-fp8", ".cache") or \
       (f.endswith(".safetensors") and f not in set(keep.values())):
        continue
    os.symlink(os.path.join(csnap, f), os.path.join(out, f))
json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
cfg = json.load(open(os.path.join(snap, "config.json"))); t = cfg.get("text_config", cfg)
t["num_experts_per_tok"] = 10
with open(os.path.join(snap, "model_extra_tensors.safetensors"), "rb") as fh:
    h = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
w = {v["shape"][0] for k, v in h.items() if k.endswith("mlp.shared_expert.gate_proj.weight")}
if len(w) == 1:
    t["shared_expert_intermediate_size"] = w.pop()
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
PY
  ssh_w "rm -rf '$CACHE_ABS/draft-k10'"
  tar -C "$CACHE_ABS" -cf - draft-k10 | ssh_w "tar -C '$CACHE_ABS' -xf -"
  F2=(); i=0
  while [ $i -lt ${#FLAGS[@]} ]; do
    f="${FLAGS[$i]}"
    case "$f" in
      --moe-backend) i=$((i + 2)); continue;;
      --speculative-config) F2+=("$f" "${FLAGS[$((i + 1))]/\{\"method\":\"mtp\",/{\"method\":\"mtp\",\"model\":\"/cache/draft-k10\",}"); i=$((i + 2)); continue;;
    esac
    F2+=("$f"); i=$((i + 1))
  done
  FLAGS=("${F2[@]}" --enable-expert-parallel)
  echo "· INT4-AutoRound: fp8 side layers, n-gram table fp8 → NVFP4 at load, its draft head (cache/draft-k10), expert parallel"
fi
# compressed-tensors checkpoints (hibrid48-uncensored: fp8 q/k/v, bf16 indexer projection) load the indexer unfused
if grep -q '"quant_method": *"compressed-tensors"' "$MODEL_DIR/config.json" 2>/dev/null; then
  ENVS+=(-e MBX_QSA_UNFUSE_INDEXER=1)
fi
compose() {  # compose <rank> <iface> <ic-ip> <hca> <has-rdma yes|no> <gid-index|"">  → prints the docker run command (quoted)
  local rank=$1 iface=$2 ic=$3 hca=$4 rdma=$5 gid=$6 a=()
  a=(docker run -d --name "$NAME" --gpus all --ipc=host --network host --cap-add SYS_PTRACE)
  [ "$rdma" = yes ] && a+=(--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1)
  [ -n "$CPUSET" ] && a+=(--cpuset-cpus "$CPUSET")
  a+=(-v "$MODELS_ABS:/models" -v "$CACHE_ABS:/cache" "${PMOUNTS[@]}"
      -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
      -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace -e VLLM_CACHE_ROOT=/cache/vllm-cache
      -e "NCCL_SOCKET_IFNAME=$iface" -e "GLOO_SOCKET_IFNAME=$iface" -e "VLLM_HOST_IP=$ic" -e NCCL_IB_DISABLE=0)
  [ -n "$hca" ] && a+=(-e "NCCL_IB_HCA=$hca")
  a+=("${ENVS[@]}")
  [ -n "$gid" ] && a+=(-e "NCCL_IB_GID_INDEX=$gid")    # probed, AFTER recipe.yaml's env → docker keeps the last -e
  a+=(--entrypoint vllm "$IMAGE" serve "/models/$LOCAL_NAME" --host "$HOST" --port "$PORT"
      --nnodes 2 --node-rank "$rank" --master-addr "$HEAD_IC" --master-port "$MPORT" --tensor-parallel-size 2)
  [ "$rank" != 0 ] && a+=(--headless)
  a+=("${FLAGS[@]}")
  printf '%q ' "${a[@]}"
}
HEAD_RDMA=$([ -d /dev/infiniband ] && echo yes || echo no)
WORKER_RDMA=$(ssh_w "[ -d /dev/infiniband ] && echo yes || echo no")
[ "$HEAD_RDMA$WORKER_RDMA" = yesyes ] || echo "  ⚠ RDMA not available on both boxes (head $HEAD_RDMA, worker $WORKER_RDMA) — running NCCL over TCP"
# ib_links: 2 → add the second PCIe half of the ConnectX-7 (an ACTIVE RDMA device other than the first, whose interface has an
# IPv4 — RoCE v2 needs one). Missing on either box → both stay on one device (a one-sided list hangs NCCL init).
SIB_PROBE='for d in /sys/class/infiniband/*; do n=${d##*/}; [ "$n" = "$1" ] && continue
  grep -q ACTIVE "$d/ports/1/state" 2>/dev/null || continue
  for nd in "$d"/device/net/*; do ip -4 -o addr show dev "${nd##*/}" 2>/dev/null | grep -q inet && { echo "$n"; exit 0; }; done; done'
if [ "$(rkey server ib_links)" = 2 ] && [ "$HEAD_RDMA$WORKER_RDMA" = yesyes ] && [ -n "$HEAD_HCA" ] && [ -n "$WORKER_HCA" ]; then
  HS="$(bash -c "$SIB_PROBE" _ "$HEAD_HCA")"; WS="$(ssh_w "bash -c $(printf '%q' "$SIB_PROBE") _ $(printf '%q' "$WORKER_HCA")")"
  if [ -n "$HS" ] && [ -n "$WS" ]; then HEAD_HCA="$HEAD_HCA,$HS"; WORKER_HCA="$WORKER_HCA,$WS"; echo "· dual rail: head $HEAD_HCA · worker $WORKER_HCA"
  else echo "  ⚠ ib_links: 2 but no second active RDMA device with an IPv4 (head '${HS:-none}', worker '${WS:-none}') — staying on one"; fi
fi
# NCCL_IB_GID_INDEX per box from the live GID table (it moves on a link flap / reboot); recipe.yaml's value is the fallback
PIN_GID="$(rkey env NCCL_IB_GID_INDEX)"; HEAD_GID=""; WORKER_GID=""
[ "$HEAD_RDMA" = yes ] && [ -n "$HEAD_HCA" ] && HEAD_GID="$(gid_index "$HEAD_HCA" "$HEAD_IFACE")"
[ "$WORKER_RDMA" = yes ] && [ -n "$WORKER_HCA" ] && WORKER_GID="$(gid_index_w "$WORKER_HCA" "$WORKER_IFACE")"
for side in head worker; do
  g=$([ $side = head ] && echo "$HEAD_GID" || echo "$WORKER_GID")
  if [ -z "$g" ]; then echo "  ⚠ $side: no RoCE v2 IPv4 GID found — using recipe.yaml's NCCL_IB_GID_INDEX=${PIN_GID:-unset}"
  elif [ -n "$PIN_GID" ] && [ "$g" != "$PIN_GID" ]; then echo "· $side: RoCE v2 GID index is $g (recipe.yaml pins $PIN_GID) — using $g"
  else echo "· $side: NCCL_IB_GID_INDEX=$g (probed)"; fi
done
compaction_check

# 4b. optional vLLM patches (recipe.yaml server.patches → patches/<name>.patch; default none): the files a patch
#     touches are copied out of the image, patched, copied to the worker and mounted read-only on both boxes.
PATCHES="$(rkey server patches)"; PMOUNTS=()
if [ -n "$PATCHES" ]; then
  command -v patch >/dev/null || { echo "✗ server.patches needs the 'patch' tool on this box (apt install patch)"; exit 1; }
  VLLM_DIR="$(docker run --rm --entrypoint python3 "$IMAGE" -c 'import importlib.util as u; print(u.find_spec("vllm").submodule_search_locations[0])')"
  STAGE="$CACHE_ABS/patched"; rm -rf "$STAGE"; mkdir -p "$STAGE"
  cid="$(docker create "$IMAGE")"; FILES=()
  pfail() { docker rm "$cid" >/dev/null 2>&1; echo "✗ $*"; exit 1; }
  for p in ${PATCHES//,/ }; do
    pf="patches/$p.patch"; [ -f "$pf" ] || pfail "server.patches: $pf not found"
    while read -r rel; do
      [ -f "$STAGE/$rel" ] && continue
      mkdir -p "$STAGE/$(dirname "$rel")"
      docker cp "$cid:$VLLM_DIR/$rel" "$STAGE/$rel" >/dev/null 2>&1 || pfail "patch $p: vllm/$rel is not in $IMAGE"
      FILES+=("$rel")
    done < <(sed -n 's#^+++ b/\([^[:space:]]*\).*#\1#p' "$pf")
    patch --dry-run -s -p1 -d "$STAGE" < "$pf" >/dev/null 2>&1 || pfail "patch $p does not fit $IMAGE — remove it from server.patches"
    patch -s -p1 --no-backup-if-mismatch -d "$STAGE" < "$pf"
    echo "· patch $p applied"
  done
  docker rm "$cid" >/dev/null
  ssh_w "rm -rf '$STAGE' && mkdir -p '$STAGE'"
  tar -C "$STAGE" -cf - . | ssh_w "tar -C '$STAGE' -xf -"
  for rel in "${FILES[@]}"; do PMOUNTS+=(-v "$STAGE/$rel:$VLLM_DIR/$rel:ro"); done
fi

# 5. launch: clear old containers, then GATE on memory (unified memory needs ~30-60 s after a container dies;
#    launching earlier = a phantom CUDA OOM), then HEAD first (the rendezvous master), then the worker — the order
#    every successful TP=2 boot of this model used; the worker retries the connect until the head listens.
docker rm -f "$NAME" >/dev/null 2>&1 || true
ssh_w "docker rm -f '$NAME' >/dev/null 2>&1 || true"
# FlashInfer's autotune cache is written by one rank only; a box holding a cache its peer lacks deadlocks the warm-up → clear both
for side in head worker; do
  c="docker run --rm --entrypoint rm -v '$CACHE_ABS:/c' '$IMAGE' -rf /c/vllm-cache/flashinfer_autotune_cache"
  if [ $side = head ]; then sh -c "$c"; else ssh_w "$c"; fi
done
[ "$(rkey env MBX_DEPTH_MODE)" = dynamic ] && echo "· draft depth: dynamic, ceiling $(rkey env MBX_DEPTH_CEIL) (recipe.yaml env)"
wait_mem 100 120 || exit 1
evict_cache "$MODEL_DIR"
echo "· starting head (rank 0) — API on $HOST:$PORT once healthy (~6 min with the weights present, ~15 min on the first boot)"
eval "$(compose 0 "$HEAD_IFACE" "$HEAD_IC" "$HEAD_HCA" "$HEAD_RDMA" "$HEAD_GID")" >/dev/null
sleep 2
echo "· starting worker ($WORKER_HOST, rank 1, --headless)"
ssh_w "$(compose 1 "$WORKER_IFACE" "$WORKER_IC" "$WORKER_HCA" "$WORKER_RDMA" "$WORKER_GID") >/dev/null"

# 6. stream the head's logs until healthy; fail fast if either container dies
echo "· streaming engine logs until healthy (Ctrl-C detaches; the cluster keeps booting)"
docker logs -f "$NAME" 2>&1 &
LOGS=$!
trap 'kill "$LOGS" 2>/dev/null || true' EXIT INT TERM
for i in $(seq 1 480); do
  if curl -sf -m 3 "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo
    echo "──────────────────────────────────────────────────────────"
    echo "✓ cluster serving — OpenAI-compatible API is live on the head"
    echo "    endpoint : http://$HOST:$PORT/v1"
    echo "    monitor  : ./view.sh          (throughput, acceptance, RDMA proof)"
    echo "    logs     : docker logs -f $NAME      · worker: ssh $WORKER docker logs -f $NAME"
    echo "    stop     : ./stop.sh          (both boxes)"
    echo "──────────────────────────────────────────────────────────"
    exit 0
  fi
  if ! docker ps -q --filter "name=^$NAME\$" | grep -q .; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo "✗ head container exited — see above. Worker's last lines:"; ssh_w "docker logs --tail 20 '$NAME'" 2>&1 | tail -20; exit 1
  fi
  if ! ssh_w "docker ps -q --filter 'name=^$NAME\$' | grep -q ." 2>/dev/null; then
    kill "$LOGS" 2>/dev/null || true; wait "$LOGS" 2>/dev/null || true
    echo "✗ worker container exited:"; ssh_w "docker logs --tail 40 '$NAME'" 2>&1 | tail -40
    docker rm -f "$NAME" >/dev/null 2>&1; exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 40 min — still booting? watch: docker logs -f $NAME"; exit 1
