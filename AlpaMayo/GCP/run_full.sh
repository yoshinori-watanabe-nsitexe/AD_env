#!/usr/bin/env bash
# Alpamayo-R1 VLA 圧縮評価 — GCP Vertex AI 実行スクリプト
#
# GCP_template/run_full.sh の7ステップ構造を踏襲:
#   [0] API 有効化 & IAM 設定
#   [1] smoke test (ローカル)
#   [2] GCS バケット作成
#   [3] Artifact Registry リポジトリ作成
#   [4] Docker ビルド & プッシュ (Cloud Build)
#   [5] Vertex AI カスタムジョブ投入
#   [6] ジョブ完了待機 & 結果ダウンロード
#
# 使用方法:
#   ./run_full.sh [PROJECT_ID] [SUFFIX]
#
# 例:
#   ./run_full.sh my-gcp-project vla-eval
#   ./run_full.sh my-gcp-project vla-eval-lora

set -euo pipefail

# ─────────────────────────────────────────────────────────────
# 設定 (GCP_template と同じ変数体系)
# ─────────────────────────────────────────────────────────────
PROJECT_ID="${1:-airy-decorator-216816}"
_SUF="${2:-vla-eval}"
REGION="asia-northeast1"
BUCKET="gs://${PROJECT_ID}-${_SUF}"
IMAGE="asia-northeast1-docker.pkg.dev/${PROJECT_ID}/${_SUF}/trainer:latest"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="${_SUF}_${TIMESTAMP}"
LOCAL_RESULT_DIR="./results/${JOB_NAME}"

# 評価パラメータ (環境変数で上書き可能)
VLA_SAMPLES="${VLA_SAMPLES:-200}"
VLA_BATCH="${VLA_BATCH:-16}"
VLA_SCALE="${VLA_SCALE:-base}"
VLA_NO_DISTILL="${VLA_NO_DISTILL:-0}"
VLA_NO_LORA="${VLA_NO_LORA:-0}"

gcloud config set project "${PROJECT_ID}"

echo "======================================================"
echo "Alpamayo-R1 VLA 評価 — Vertex AI 実行スクリプト"
echo "======================================================"
echo "  PROJECT_ID  : ${PROJECT_ID}"
echo "  SUFFIX      : ${_SUF}"
echo "  REGION      : ${REGION}"
echo "  IMAGE       : ${IMAGE}"
echo "  BUCKET      : ${BUCKET}/results/${JOB_NAME}"
echo "  VLA_SAMPLES : ${VLA_SAMPLES}"
echo "  VLA_SCALE   : ${VLA_SCALE}"
echo "======================================================"

# setting.yaml を envsubst で生成 (GCP_template と同じ方法)
export IMAGE BUCKET JOB_NAME
export VLA_SAMPLES VLA_BATCH VLA_SCALE VLA_NO_DISTILL VLA_NO_LORA
envsubst < setting.yaml.template > setting.yaml

# ─────────────────────────────────────────────────────────────
# [0] API 有効化 & IAM 設定 (初回のみ・冪等)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [0] API 有効化..."
gcloud services enable \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    aiplatform.googleapis.com \
    --project="${PROJECT_ID}"

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo ">>> [0b] Cloud Build SA (${CB_SA}) に権限付与..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CB_SA}" \
    --role="roles/artifactregistry.writer" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# [1] smoke test (ローカル実行)
#     本スクリプト: python evaluate.py --scale tiny (高速モック)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [1] smoke test (ローカル, --scale tiny)..."
SMOKE_OUT="./smoke_result"
mkdir -p "${SMOKE_OUT}"

python evaluate.py \
    --samples 20 \
    --batch   4 \
    --scale   tiny \
    --no-distill \
    --no-lora \
    2>&1 | tee "${SMOKE_OUT}/smoke.log"

# smoke test 失敗チェック
if grep -qiE "error|traceback|exception" "${SMOKE_OUT}/smoke.log" 2>/dev/null; then
    echo "[ERROR] smoke test でエラーが検出されました。中断します。"
    cat "${SMOKE_OUT}/smoke.log"
    exit 1
fi
echo "[OK] smoke test 通過"

# ─────────────────────────────────────────────────────────────
# [2] GCS バケット作成 (初回のみ)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [2] GCS バケット作成..."
gsutil mb -l "${REGION}" "${BUCKET}" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# [3] Artifact Registry リポジトリ作成 (初回のみ)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [3] Artifact Registry リポジトリ作成..."
gcloud artifacts repositories create "${_SUF}" \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" 2>/dev/null || true

# ─────────────────────────────────────────────────────────────
# [4] Docker イメージをビルド & プッシュ (Cloud Build)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [4] Docker ビルド & プッシュ..."
gcloud builds submit \
    --config cloudbuild.yaml \
    --project="${PROJECT_ID}" \
    --substitutions="_SUF=${_SUF}"

# ─────────────────────────────────────────────────────────────
# [5] Vertex AI カスタムジョブ投入
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [5] Vertex AI ジョブ投入: ${JOB_NAME}"
gcloud ai custom-jobs create \
    --region="${REGION}" \
    --display-name="${JOB_NAME}" \
    --config="setting.yaml"

echo ""
echo "ジョブ投入完了: ${JOB_NAME}"
echo "結果保存先   : ${BUCKET}/results/${JOB_NAME}/"

# ─────────────────────────────────────────────────────────────
# [6] ジョブ完了待機 & 結果ダウンロード
#     GCP_template と同じポーリングロジック (60秒間隔)
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [6] ジョブ完了待機 (60秒間隔でポーリング)..."

WAIT_SECONDS=0
MAX_WAIT=14400  # 最大4時間

while [ "${WAIT_SECONDS}" -lt "${MAX_WAIT}" ]; do
    STATUS=$(gcloud ai custom-jobs list \
        --region="${REGION}" \
        --filter="displayName=${JOB_NAME}" \
        --format="value(state)" 2>/dev/null | head -1)

    echo "  ステータス: ${STATUS}  (経過: ${WAIT_SECONDS}s / $(date '+%H:%M:%S'))"

    case "${STATUS}" in
        JOB_STATE_SUCCEEDED)
            echo "[OK] ジョブ完了"
            break
            ;;
        JOB_STATE_FAILED|JOB_STATE_CANCELLED)
            echo "[ERROR] ジョブが失敗またはキャンセルされました: ${STATUS}"
            # ログを取得して表示
            gcloud ai custom-jobs describe \
                "$(gcloud ai custom-jobs list \
                    --region="${REGION}" \
                    --filter="displayName=${JOB_NAME}" \
                    --format='value(name)' | head -1)" \
                --region="${REGION}" 2>/dev/null || true
            exit 1
            ;;
        *)
            sleep 60
            WAIT_SECONDS=$((WAIT_SECONDS + 60))
            ;;
    esac
done

if [ "${WAIT_SECONDS}" -ge "${MAX_WAIT}" ]; then
    echo "[ERROR] タイムアウト (${MAX_WAIT}秒) — ジョブを確認してください"
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# [7] 結果ダウンロード
# ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [7] 結果ダウンロード → ${LOCAL_RESULT_DIR}"
mkdir -p "${LOCAL_RESULT_DIR}"
gsutil -m cp -r "${BUCKET}/results/${JOB_NAME}/*" "${LOCAL_RESULT_DIR}/" 2>/dev/null || true

echo ""
echo "===== 完了 ====="
echo "ローカル結果: ${LOCAL_RESULT_DIR}"
ls -lh "${LOCAL_RESULT_DIR}" 2>/dev/null || echo "(ファイルなし)"

# JSON サマリーがあれば表示
RESULT_JSON="${LOCAL_RESULT_DIR}/evaluation_results.json"
if [ -f "${RESULT_JSON}" ]; then
    echo ""
    echo "--- 評価結果サマリー ---"
    python3 - <<'PYEOF'
import json, sys, os
path = os.environ.get("RESULT_JSON", "")
if not path or not os.path.exists(path):
    # 最新の results/ を探す
    import glob
    files = sorted(glob.glob("results/*/evaluation_results.json"))
    path = files[-1] if files else ""
if not path:
    print("(結果 JSON が見つかりません)")
    sys.exit(0)
with open(path) as f:
    data = json.load(f)
print(f"{'手法':<22} {'L2':>8} {'Acc':>8} {'Mem(MB)':>9}")
print("-" * 52)
for r in data:
    m = r["metrics"]
    print(f"{r['method']:<22} {m.get('planning_l2',0):>8.4f} "
          f"{m.get('action_accuracy',0):>8.3f} "
          f"{r['info'].get('memory_mb',0):>9.1f}")
PYEOF
fi
