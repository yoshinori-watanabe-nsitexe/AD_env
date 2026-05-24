"""
trainer/task.py
Vertex AI カスタムジョブのエントリーポイント

GCP_template の trainer/task.py パターンを踏襲:
  - 環境変数 AIP_MODEL_DIR から GCS 出力先を取得
  - evaluate.main() を実行
  - 結果 JSON を GCS にアップロード

環境変数 (Vertex AI が自動設定 + setting.yaml.template で追加):
  AIP_MODEL_DIR    : gs://bucket/results/job_name  (必須)
  VLA_SAMPLES      : テストサンプル数 (デフォルト 200)
  VLA_BATCH        : バッチサイズ   (デフォルト 16)
  VLA_SCALE        : モデルスケール "base" | "tiny" (デフォルト base)
  VLA_NO_DISTILL   : "1" で蒸留スキップ
  VLA_NO_LORA      : "1" で LoRA スキップ
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

# ─── GCS クライアント ────────────────────────────────────────
try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    print("[WARN] google-cloud-storage 未インストール: GCS アップロードをスキップ")


def upload_dir_to_gcs(local_dir: Path, gcs_uri: str) -> None:
    """ローカルディレクトリを GCS にアップロード"""
    if not HAS_GCS:
        print(f"[SKIP] GCS upload: {local_dir} → {gcs_uri}")
        return

    # gs://bucket/path/to/dir → bucket, path/to/dir
    assert gcs_uri.startswith("gs://"), f"Invalid GCS URI: {gcs_uri}"
    parts = gcs_uri[5:].split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    client = gcs.Client()
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for fpath in sorted(local_dir.rglob("*")):
        if fpath.is_file() and not fpath.name.startswith("."):
            blob_name = f"{prefix}/{fpath.relative_to(local_dir)}" if prefix else str(fpath.relative_to(local_dir))
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(fpath))
            print(f"  uploaded: {blob_name}")
            uploaded += 1

    print(f"[GCS] {uploaded} ファイルを {gcs_uri} にアップロード完了")


def main() -> None:
    # ─── 環境変数から設定取得 ─────────────────────────────────
    aip_model_dir = os.environ.get("AIP_MODEL_DIR", "")
    num_samples   = int(os.environ.get("VLA_SAMPLES",    "200"))
    batch_size    = int(os.environ.get("VLA_BATCH",      "16"))
    scale         = os.environ.get("VLA_SCALE",          "base")
    no_distill    = os.environ.get("VLA_NO_DISTILL",     "0") == "1"
    no_lora       = os.environ.get("VLA_NO_LORA",        "0") == "1"

    print("=" * 60)
    print("Alpamayo-R1 VLA 圧縮評価 — Vertex AI ジョブ")
    print("=" * 60)
    print(f"  AIP_MODEL_DIR : {aip_model_dir or '(ローカル実行)'}")
    print(f"  VLA_SAMPLES   : {num_samples}")
    print(f"  VLA_BATCH     : {batch_size}")
    print(f"  VLA_SCALE     : {scale}")
    print(f"  NO_DISTILL    : {no_distill}")
    print(f"  NO_LORA       : {no_lora}")
    print("=" * 60)

    # ─── 評価実行 ────────────────────────────────────────────
    # sys.path に /app を追加 (コンテナ内パス)
    app_dir = Path(__file__).parent.parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

    from evaluate import main as eval_main  # noqa: E402

    try:
        results = eval_main(
            num_test_samples=num_samples,
            batch_size=batch_size,
            scale=scale,
            run_distillation=not no_distill,
            run_lora=not no_lora,
        )
        exit_code = 0
    except Exception:
        traceback.print_exc()
        results = None
        exit_code = 1

    # ─── ローカル結果の確認 ──────────────────────────────────
    local_results = app_dir / "results"
    result_json   = local_results / "evaluation_results.json"

    if result_json.exists():
        with open(result_json) as f:
            data = json.load(f)
        print(f"\n[結果サマリー] {len(data)} 手法の評価完了")
        for r in data:
            m = r["metrics"]
            print(f"  {r['method']:<22} L2={m.get('planning_l2', float('nan')):.4f}"
                  f"  Acc={m.get('action_accuracy', float('nan')):.3f}"
                  f"  Mem={r['info'].get('memory_mb', 0):.1f}MB")
    else:
        print("[WARN] result JSON が見つかりません")

    # ─── GCS アップロード ────────────────────────────────────
    if aip_model_dir and local_results.exists():
        print(f"\n[GCS] 結果をアップロード: {aip_model_dir}")
        upload_dir_to_gcs(local_results, aip_model_dir)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
