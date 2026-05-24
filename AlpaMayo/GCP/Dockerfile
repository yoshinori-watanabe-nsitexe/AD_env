# Alpamayo-R1 VLA 圧縮評価 — Vertex AI カスタムジョブ用イメージ
#
# ベース: GCP_template/Dockerfile の構造を踏襲
#   pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
#   → VLA評価に必要なパッケージを追加
#
# 本番 Alpamayo-R1-10B を動かす場合は VRAM 24GB+ 必須 (A100 × 1 以上)
# モックモデルによる CI/smoke test は CPU のみで動作する

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# ─── システム依存 ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# ─── Python 依存パッケージ ──────────────────────────────────
# peft: LoRA / PEFT
# accelerate: multi-GPU / Vertex AI 推奨
# google-cloud-storage: GCS への結果書き込み
# torchvision: 画像前処理ユーティリティ
# scikit-learn: メトリクス計算補助
RUN pip install --no-cache-dir \
    torchvision==0.17.0 \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    accelerate>=0.28.0 \
    google-cloud-storage>=2.14.0 \
    scikit-learn>=1.4.0 \
    tqdm>=4.66.0

# ─── アプリケーションコード ─────────────────────────────────
# GCP_template と同じく trainer/ をモジュールとして配置
COPY vla_model.py   .
COPY vla_dataset.py .
COPY evaluate.py    .
COPY trainer/       trainer/

# 結果キャッシュ用ディレクトリ (GCS 書き込み前の一時置き場)
RUN mkdir -p /app/results

# ─── エントリーポイント ─────────────────────────────────────
# GCP_template: ENTRYPOINT ["python", "-m", "trainer.task"]
# 本スクリプト: trainer/task.py が evaluate.py を呼び出し
ENTRYPOINT ["python", "-m", "trainer.task"]
