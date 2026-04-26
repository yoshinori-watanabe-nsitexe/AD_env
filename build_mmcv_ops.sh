#!/bin/bash
# ============================================================
# build_mmcv_ops.sh
# mmcv 1.6.2 の CUDA op を初回コンテナ起動後にビルドするスクリプト
#
# 実行タイミング:
#   docker build 時は GPU が存在しないため CUDA op をビルドできない。
#   このスクリプトをコンテナ起動後に一度だけ実行することで
#   CUDA op を有効化する。
#
# 使い方:
#   docker exec -it uniad2      build_mmcv_ops.sh
#   docker exec -it algengine   build_mmcv_ops.sh
#
# または Makefile から:
#   make build-mmcv-ops-uniad2
#   make build-mmcv-ops-algengine
# ============================================================
set -e

MMCV_SRC=/build/mmcv

echo "================================================"
echo " mmcv CUDA op ビルド開始"
echo " ソースディレクトリ: ${MMCV_SRC}"
echo "================================================"

# GPU の存在確認
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No GPU'" 2>/dev/null; then
    echo "[ERROR] GPU が検出されません。GPU を割り当ててコンテナを起動してください。"
    echo "        例: docker run --gpus all ..."
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
echo "[INFO] 検出された GPU: ${GPU_NAME}"

# すでにビルド済みか確認
if python3 -c "from mmcv.ops import get_compiling_cuda_version" 2>/dev/null; then
    CUDA_VER=$(python3 -c "from mmcv.ops import get_compiling_cuda_version; print(get_compiling_cuda_version())")
    echo "[INFO] mmcv CUDA op は既にビルド済みです (CUDA ${CUDA_VER})。スキップします。"
    echo "       再ビルドする場合は --force オプションを付けてください。"
    if [ "$1" != "--force" ]; then
        exit 0
    fi
    echo "[INFO] --force が指定されました。再ビルドします。"
fi

# CUDA op ビルド
echo "[INFO] CUDA op をビルドします..."
cd "${MMCV_SRC}"

MMCV_WITH_OPS=1 python setup.py build_ext --inplace 2>&1 | tail -20

echo ""
echo "================================================"
echo " ビルド完了。動作確認中..."
echo "================================================"

# 動作確認
python3 - <<'VERIFY'
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(f"  CUDA バージョン : {get_compiling_cuda_version()}")
print(f"  コンパイラ      : {get_compiler_version()}")

from mmcv.ops import nms, roi_align
print("  nms       : OK")
print("  roi_align : OK")

import torch
x = torch.randn(4, 4).cuda()
print(f"  GPU テンソル演算: OK ({x.device})")
VERIFY

echo ""
echo "================================================"
echo " mmcv CUDA op のビルドが完了しました。"
echo "================================================"
