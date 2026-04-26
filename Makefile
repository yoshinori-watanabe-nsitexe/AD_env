# ============================================================
# Makefile — UniAD + WorldEngine Docker ビルド・運用
#
# ビルド順序:
#   ① base       → 共有レイヤ (APT / Python / PyTorch)
#   ② uniad2     → base を継承 (mmcv wheel + UniAD)
#   ③ algengine  → base を継承 (mmcv src build + WorldEngine)
#   ④ simengine  → base を継承 (gsplat + Ray + SimEngine)
#
# ② ③ ④ は base さえできれば並列ビルド可能
# ============================================================
.PHONY: help build-base build-uniad2 build-algengine build-simengine \
        build-all build-parallel \
        up-uniad2 up-algengine up-simengine up-all \
        train-stage1 train-stage2 train-rl \
        eval-openloop eval-closedloop \
        rollout extract-rare tensorboard \
        down clean

COMPOSE   = docker compose
ENV_FILE  = .env
BASE_TAG  = uniad-worldengine-base:latest

help:
	@echo "=== UniAD + WorldEngine Docker ==="
	@echo ""
	@echo "【ビルド】"
	@echo "  make build-base       1 ベースイメージ (必ず最初に実行)"
	@echo "  make build-all        1〜4 を順番にビルド"
	@echo "  make build-parallel   1の後に2,3,4を並列ビルド (make -j3)"
	@echo "  make build-uniad2     2 UniAD 事前学習イメージ"
	@echo "  make build-algengine  3 AlgEngine RL イメージ"
	@echo "  make build-simengine  4 SimEngine 3DGS イメージ"
	@echo ""
	@echo "【起動】"
	@echo "  make up-uniad2        UniAD コンテナ起動"
	@echo "  make up-algengine     AlgEngine コンテナ起動"
	@echo "  make up-simengine     SimEngine HEAD + WORKER 起動"
	@echo "  make tensorboard      TensorBoard (port 6006)"
	@echo "  make build-mmcv-ops-uniad2    mmcv CUDA op ビルド (uniad2コンテナ)"
	@echo "  make build-mmcv-ops-algengine mmcv CUDA op ビルド (algengineコンテナ)"
	@echo ""
	@echo "【学習・評価】"
	@echo "  make train-stage1     UniAD Stage1 Perception 学習"
	@echo "  make train-stage2     UniAD Stage2 E2E 学習"
	@echo "  make train-rl         AlgEngine RL ファインチューニング"
	@echo "  make rollout          SimEngine ロールアウト生成"
	@echo "  make extract-rare     希少ケース抽出"
	@echo "  make eval-openloop    Open-loop 評価"
	@echo "  make eval-closedloop  Closed-loop 評価"
	@echo ""
	@echo "【クリーンアップ】"
	@echo "  make down             全コンテナ停止"
	@echo "  make clean            コンテナ・イメージを削除"

# ────────────────────────────────────────────────────────────
# ビルド — 順序が重要
# ────────────────────────────────────────────────────────────

# ① ベースイメージ (他の全イメージが依存するため必ず最初に実行)
build-base:
	docker build \
	    -f Dockerfile.base \
	    -t $(BASE_TAG) \
	    .
	@echo "✓ base image built: $(BASE_TAG)"

# ② UniAD 事前学習 (build-base が完了してから実行)
build-uniad2: build-base
	docker build \
	    -f Dockerfile.uniad2 \
	    -t uniad2:latest \
	    .

# ③ AlgEngine RL (build-base が完了してから実行)
build-algengine: build-base
	docker build \
	    -f Dockerfile.algengine \
	    -t algengine:latest \
	    .

# ④ SimEngine 3DGS (build-base が完了してから実行)
build-simengine: build-base
	docker build \
	    -f Dockerfile.simengine \
	    -t simengine:latest \
	    .

# 全イメージを順番にビルド (安全な直列実行)
build-all: build-base build-uniad2 build-algengine build-simengine
	@echo "✓ All images built."

# ② ③ ④ を並列ビルド (base 完了後に make -j3 で実行)
# 使い方: make build-base && make build-parallel -j3
build-parallel: build-uniad2 build-algengine build-simengine

# ────────────────────────────────────────────────────────────
# 起動
# ────────────────────────────────────────────────────────────
up-uniad2:
	$(COMPOSE) --env-file $(ENV_FILE) up -d uniad2

up-algengine:
	$(COMPOSE) --env-file $(ENV_FILE) up -d algengine

up-simengine:
	$(COMPOSE) --env-file $(ENV_FILE) up -d simengine-head simengine-worker

up-all:
	$(COMPOSE) --env-file $(ENV_FILE) up -d

tensorboard:
	$(COMPOSE) --env-file $(ENV_FILE) up -d tensorboard
	@echo "TensorBoard: http://localhost:6006"

scale-workers:
	$(COMPOSE) --env-file $(ENV_FILE) \
	    up -d --scale simengine-worker=$(N) simengine-worker

# ────────────────────────────────────────────────────────────
# 学習・評価
# ────────────────────────────────────────────────────────────
train-stage1:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    bash -c "cd /workspace/UniAD && \
	    ./tools/dist_train.sh \
	        projects/configs/stage1_track_map/base_track_map.py 8 \
	        --work-dir work_dirs/stage1_base"

train-stage2:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    bash -c "cd /workspace/UniAD && \
	    ./tools/dist_train.sh \
	        projects/configs/stage2_e2e/base_e2e.py 8 \
	        --work-dir work_dirs/stage2_e2e"

train-rl:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    bash -c "./scripts/e2e_dist_train.sh \
	        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py 8 \
	        work_dirs/e2e_uniad_50pct/epoch_20.pth"

rollout:
	$(COMPOSE) --env-file $(ENV_FILE) exec simengine-head \
	    bash scripts/run_ray_distributed_rollout.sh \
	        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct.py \
	        /workspace/WorldEngine/data/alg_engine/ckpts/e2e_uniad_50pct_ep20.pth \
	        e2e_uniad_50pct \
	        navtrain_50pct_collision \
	        navtrain

extract-rare:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    python scripts/rare_case_sampling_by_pdms.py \
	        --pdm-result work_dirs/e2e_uniad_50pct/navtest.csv \
	        --base-split configs/navsim_splits/navtest_split/navtest.yaml \
	        --output-dir configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare

eval-openloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    bash -c "./scripts/e2e_dist_eval.sh \
	        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth 8"

eval-closedloop:
	$(COMPOSE) --env-file $(ENV_FILE) exec simengine-head \
	    bash /workspace/WorldEngine/projects/AlgEngine/scripts/run_ray_distributed_testing.sh \
	        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
	        /workspace/WorldEngine/projects/AlgEngine/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
	        e2e_uniad_rlft navtest_failures NR

# ────────────────────────────────────────────────────────────
# クリーンアップ
# ────────────────────────────────────────────────────────────
down:
	$(COMPOSE) --env-file $(ENV_FILE) down

clean:
	$(COMPOSE) --env-file $(ENV_FILE) down --rmi local --volumes --remove-orphans
	docker rmi $(BASE_TAG) 2>/dev/null || true

# ────────────────────────────────────────────────────────────
# mmcv CUDA op 後ビルド (初回コンテナ起動後に一度だけ実行)
# ────────────────────────────────────────────────────────────
build-mmcv-ops-uniad2:
	$(COMPOSE) --env-file $(ENV_FILE) exec uniad2 \
	    build_mmcv_ops.sh

build-mmcv-ops-algengine:
	$(COMPOSE) --env-file $(ENV_FILE) exec algengine \
	    build_mmcv_ops.sh

build-mmcv-ops-all: build-mmcv-ops-uniad2 build-mmcv-ops-algengine
	@echo "✓ mmcv CUDA ops built in all containers."
