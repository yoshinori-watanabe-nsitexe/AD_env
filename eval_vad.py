"""
eval_scripts/eval_vad.py
VAD (VAD-Tiny / VAD-Base / VADv2) 評価スクリプト

VAD の評価は必ず 1 GPU で実行する必要がある。
(公式: multi-GPU 評価は結果が不正確になる)

使い方:
    python eval_scripts/eval_vad.py \
        --config  projects/configs/VAD/VAD_base_stage_2.py \
        --ckpt    ckpts/VAD_base.pth \
        --out-dir eval_results/vad_base \
        --variant base

    # VADv2
    python eval_scripts/eval_vad.py \
        --config  VADv2/configs/vadv2_stage2.py \
        --ckpt    ckpts/VADv2.pth \
        --variant v2
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from eval_base import get_logger, REFERENCE


# VAD 参照値 (nuScenes val, open-loop)
VAD_REFERENCE = {
    "VAD-Tiny": {
        "avg_L2":   0.78, "avg_Col":  0.38,
        "L2_1s":    0.46, "L2_2s":    0.76, "L2_3s":   1.12,
        "Col_1s":   0.21, "Col_2s":   0.35, "Col_3s":  0.58,
        "FPS":      16.8,
    },
    "VAD-Base": {
        "avg_L2":   0.72, "avg_Col":  0.22,
        "L2_1s":    0.41, "L2_2s":    0.70, "L2_3s":   1.05,
        "Col_1s":   0.07, "Col_2s":   0.17, "Col_3s":  0.41,
        "FPS":      4.5,
    },
}

# UniAD 参照値 (比較用)
UNIAD_REFERENCE = {
    "avg_L2":  0.708, "avg_Col": 0.290,
    "L2_1s":   0.236, "L2_2s":  0.599, "L2_3s":  1.289,
    "Col_1s":  0.002, "Col_2s": 0.128, "Col_3s": 0.741,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VAD 評価スクリプト")
    p.add_argument("--config",   required=True)
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--out-dir",  default="eval_results/vad")
    p.add_argument("--variant",  default="base",
                   choices=["tiny", "base", "v2"],
                   help="モデルバリアント (tiny/base/v2)")
    p.add_argument("--ann-file", default="",
                   help="カスタム ann_file (デフォルト: config の設定を使用)")
    p.add_argument("--vad-root", default="/workspace/VAD",
                   help="VAD リポジトリルート")
    return p.parse_args()


# ログからの指標抽出パターン
METRIC_PATTERNS = {
    "L2_1s":   r"L2_1\s+([\d.]+)",
    "L2_2s":   r"L2_2\s+([\d.]+)",
    "L2_3s":   r"L2_3\s+([\d.]+)",
    "Col_1s":  r"Col_1\s+([\d.]+)",
    "Col_2s":  r"Col_2\s+([\d.]+)",
    "Col_3s":  r"Col_3\s+([\d.]+)",
}


def run_vad_eval(config: str, ckpt: str, out_dir: Path,
                 ann_file: str, vad_root: str) -> str:
    """VAD の評価を実行してログを返す。"""
    result_pkl = out_dir / "vad_results.pkl"
    log_file   = out_dir / "inference.log"

    # VAD は単一 GPU 評価のみ有効
    cmd = [
        "python3", "tools/test.py",
        config,
        ckpt,
        "--launcher", "none",
        "--eval", "bbox",
        "--tmpdir", str(out_dir / "tmp"),
        "--out", str(result_pkl),
    ]
    if ann_file:
        cmd += ["--cfg-options", f"data.test.ann_file={ann_file}"]

    with log_file.open("w") as lf:
        proc = subprocess.run(
            cmd,
            cwd=vad_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lf.write(proc.stdout)
        print(proc.stdout, end="")

    if proc.returncode != 0:
        raise RuntimeError(f"VAD 評価失敗 (exit={proc.returncode})")

    return proc.stdout


def extract_metrics(log_text: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for key, pat in METRIC_PATTERNS.items():
        m = re.search(pat, log_text)
        metrics[key] = float(m.group(1)) if m else None

    # 平均値を計算
    l2_vals  = [metrics[f"L2_{t}s"]  for t in [1,2,3]
                if metrics.get(f"L2_{t}s")  is not None]
    col_vals = [metrics[f"Col_{t}s"] for t in [1,2,3]
                if metrics.get(f"Col_{t}s") is not None]
    metrics["avg_L2"]  = sum(l2_vals)  / len(l2_vals)  if l2_vals  else None
    metrics["avg_Col"] = sum(col_vals) / len(col_vals) if col_vals else None
    return metrics


def print_comparison(
    metrics: dict[str, float | None],
    variant: str,
) -> None:
    ref_key = f"VAD-{variant.capitalize()}"
    vad_ref  = VAD_REFERENCE.get(ref_key, {})
    uniad_ref = UNIAD_REFERENCE

    print(f"\n{'='*62}")
    print(f"  VAD-{variant.upper()} 評価結果 vs 参照値")
    print(f"{'='*62}")
    print(f"  {'指標':<12} {'本モデル':>10} {'VAD ref':>10} {'UniAD ref':>10}")
    print(f"  {'-'*56}")

    for t in [1, 2, 3]:
        l2  = metrics.get(f"L2_{t}s")
        col = metrics.get(f"Col_{t}s")
        l2_str  = f"{l2:.4f}"  if l2  is not None else "N/A"
        col_str = f"{col:.4f}" if col is not None else "N/A"

        vad_l2  = vad_ref.get(f"L2_{t}s",  "—")
        vad_col = vad_ref.get(f"Col_{t}s", "—")
        u_l2    = uniad_ref.get(f"L2_{t}s",  "—")
        u_col   = uniad_ref.get(f"Col_{t}s", "—")

        print(f"  L2  @{t}s      {l2_str:>10} {str(vad_l2):>10} {str(u_l2):>10}")
        print(f"  Col @{t}s      {col_str:>10} {str(vad_col):>10} {str(u_col):>10}")

    print(f"  {'-'*56}")
    avg_l2  = metrics.get("avg_L2")
    avg_col = metrics.get("avg_Col")
    print(f"  avg.L2  ↓   {str(f'{avg_l2:.4f}' if avg_l2 else 'N/A'):>10}"
          f" {str(vad_ref.get('avg_L2','—')):>10} {str(uniad_ref.get('avg_L2','—')):>10}")
    print(f"  avg.Col ↓   {str(f'{avg_col:.4f}' if avg_col else 'N/A'):>10}"
          f" {str(vad_ref.get('avg_Col','—')):>10} {str(uniad_ref.get('avg_Col','—')):>10}")
    print(f"{'='*62}")
    print(f"  推論速度参照: VAD-Tiny {VAD_REFERENCE['VAD-Tiny']['FPS']} FPS"
          f" / VAD-Base {VAD_REFERENCE['VAD-Base']['FPS']} FPS / UniAD ~1.8 FPS")
    print(f"{'='*62}")


def main() -> int:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger  = get_logger("eval_vad", out_dir / "eval_vad.log")

    logger.info("=" * 56)
    logger.info(f" VAD 評価開始: {args.variant.upper()}")
    logger.info(f" Config : {args.config}")
    logger.info(f" Ckpt   : {args.ckpt}")
    logger.info("=" * 56)
    logger.info("[注意] VAD の評価は 1 GPU で実行します (multi-GPU は不正確)")

    try:
        log_text = run_vad_eval(
            args.config, args.ckpt, out_dir,
            args.ann_file, args.vad_root,
        )
        metrics = extract_metrics(log_text)

        # ログから取れない場合は保存済みログを再読み込み
        if all(v is None for v in metrics.values()):
            saved = out_dir / "inference.log"
            if saved.exists():
                metrics = extract_metrics(saved.read_text())

        print_comparison(metrics, args.variant)

        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )
        logger.info(f"結果保存: {out_dir}/metrics.json")
        logger.info(" VAD 評価完了")
        return 0

    except Exception as e:
        logger.error(f"評価失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
