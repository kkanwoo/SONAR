#!/usr/bin/env bash
set -euo pipefail

# ========= Common Paths (Based on your environment) =========
export ROOT=~/SONAR

# WebQA+MMQA (Original, Clean) Artifacts: Assumed to be already created
IDS=$ROOT/results/ids.json
EMB=$ROOT/results/embeds.npy
INDEX_PARQUET=$ROOT/data/index.parquet

# Reuse trained codebook
CODEBOOK=$ROOT/results/ivfpq_trained.index

# ========= Fixed Hyperparameters =========
M=16                   # <<< Experiment point: Changed to 16
ALPHA=0.80
TCOS=0.75
TFRAC=0.1
SD=16
EPS=12
NORM=linf
STEPS=100
STEP_SIZE=1.0
MIX_BASE=mu
MU_MOM=0.9
BATCH=8
DEVICE=cuda
MODEL="ViT-L-14"
PRETRAINED="openai"
SECRET="cvpr2026-sonar-secret-key-v1"

# Probe Generation (Bank-aware image probes must be regenerated if M changes)
PROBE_STEPS=600
PROBE_LR=0.05
PROBE_TV=1e-4
PROBE_INIT=zeros
PROBE_SEED=1234
PROBE_TARGET_COS=0.80

# Evaluation Common
KLIST=(1 5 10 20)
WSN_THRESH=2
SUCC_THRESH=1   # Reflecting your recent settings
TAG="bank${M}_a080_tc075_tf010_eps${EPS}_sd${SD}_n${NORM}_st${STEPS}_ss${STEP_SIZE}"

# ========= Output Root (Organized in new folder) =========
RUNROOT=$ROOT/ablation_M_webmmqa/$TAG
LOGDIR=$RUNROOT/logs
mkdir -p "$RUNROOT" "$LOGDIR" "$ROOT/results/wm_ids"

echo
echo "====================================================================="
echo "[RUN WEBMMQA] M=${M} → TAG=${TAG}"
echo "  RUNROOT: $RUNROOT"
echo "====================================================================="

# 1) Watermark Injection (Dataset Filter: WebQA, MMQA)
python "$ROOT/beacon/sonar_watermark.py" \
  --index-parquet "$INDEX_PARQUET" \
  --ids-json      "$IDS" \
  --images-root   "$ROOT/data" \
  --out-dir       "$ROOT/beacon/out_${TAG}" \
  --secret-seed   "$SECRET" \
  --subspace-dim  $SD \
  --beacon-bank   $M \
  --embeds-path   "$EMB" \
  --mix-base      "$MIX_BASE" \
  --target-frac   $TFRAC \
  --norm          "$NORM" \
  --eps           $EPS \
  --steps         $STEPS \
  --step-size     $STEP_SIZE \
  --alpha         $ALPHA \
  --target-cos    $TCOS \
  --mu-momentum   $MU_MOM \
  --cosine-decay \
  --batch         $BATCH \
  --device        "$DEVICE" \
  --dataset-filter "WebQA,MMQA" \
  2>&1 | tee "$LOGDIR/01_injection.log"

# 2) Watermark ID List
awk '{print}' "$ROOT/beacon/out_${TAG}/injection_meta.jsonl" \
  | python -c 'import sys,json;print("\n".join({str(json.loads(l)["id"]) for l in sys.stdin if l.strip()}))' \
  > "$ROOT/results/wm_ids/wm_ids_${TAG}.txt"
WM_TXT="$ROOT/results/wm_ids/wm_ids_${TAG}.txt"

# 3) Re-embedding (Reflecting Watermark)
# Fixed typo: make_embeds_watermakred.py -> make_embeds_watermark.py
python "$ROOT/retriever/make_embeds_watermark.py" \
  --ids        "$IDS" \
  --embeds     "$EMB" \
  --meta       "$ROOT/beacon/out_${TAG}/injection_meta.jsonl" \
  --images-dir "$ROOT/beacon/out_${TAG}/images" \
  --out        "$RUNROOT/embeds_webmmqa_wm_${TAG}.npy" \
  --model "$MODEL" --pretrained "$PRETRAINED" --device "$DEVICE" --batch 128 \
  2>&1 | tee "$LOGDIR/02_reembed.log"

# 4) FAISS (Index reflecting Watermark, reusing Codebook)
python "$ROOT/retriever/faiss_index.py" \
  --config "$ROOT/configs/retriever.yaml" \
  --embeds "$RUNROOT/embeds_webmmqa_wm_${TAG}.npy" \
  --out    "$RUNROOT/faiss_webmmqa_${TAG}.index" \
  --reuse-index "$CODEBOOK" \
  2>&1 | tee "$LOGDIR/03_faiss.log"
FAISS_WM="$RUNROOT/faiss_webmmqa_${TAG}.index"

# 5) (Important) Generate Bank-aware Image Probes (Matching M=16!)
PROBE_OUT="$RUNROOT/image_probes_${TAG}"
python "$ROOT/retriever/make_image_probes.py" \
  --secret-seed "$SECRET" \
  --subspace-dim $SD \
  --embeds-path  "$EMB" \
  --alpha-hat    $ALPHA \
  --beacon-bank  $M \
  --out-dir      "$PROBE_OUT" \
  --model        "$MODEL" \
  --pretrained   "$PRETRAINED" \
  --device       "$DEVICE" \
  --steps        $PROBE_STEPS \
  --lr           $PROBE_LR \
  --target-cos   $PROBE_TARGET_COS \
  --tv-lambda    $PROBE_TV \
  --init         $PROBE_INIT \
  --seed         $PROBE_SEED \
  2>&1 | tee "$LOGDIR/04_probes.log"

PROBE_DIR="$PROBE_OUT/images"
PROBE_META="$PROBE_OUT/image_probe_meta.jsonl"

# 6-a) Evaluation: Bank-aware Probes (Reflecting your recent run_eval options)
for K in "${KLIST[@]}"; do
  python -m eval.run_eval \
    --query-images  "$PROBE_DIR" \
    --ids           "$IDS" \
    --faiss         "$FAISS_WM" \
    --out-metrics   "$RUNROOT/metrics_bankaware_k${K}.csv" \
    --out-topk      "$RUNROOT/retrieval_topk_bankaware_k${K}.jsonl" \
    --q-embeds      "$RUNROOT/q_embeds_bankaware.npy" \
    --mode wm --wm-ids "$WM_TXT" \
    --topk $K \
    --bank-aware \
    --wm-map    "$ROOT/beacon/out_${TAG}/injection_meta.jsonl" \
    --probe-meta "$PROBE_META" \
    --wsn-thresh $WSN_THRESH \
    --succ-thresh $SUCC_THRESH \
    --ndcg-k $K \
    2>&1 | tee "$LOGDIR/05_eval_bankaware_k${K}.log"
done

echo "[DONE] WEBMMQA M=${M} Complete → $RUNROOT"
