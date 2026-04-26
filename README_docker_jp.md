# UniAD v2.0 + WorldEngine — Docker 環境 README

End-to-End 自動運転モデル (UniAD v2.0) と世界モデルベース RL ポストトレーニング (WorldEngine) を  
Docker コンテナで完結させるためのビルド・評価手順書です。

---

## ファイル構成

```
.
├── Dockerfile.base                  # 共有ベースイメージ (必ず最初にビルド)
├── Dockerfile.uniad2                # UniAD v2.0 事前学習環境
├── Dockerfile.algengine             # WorldEngine AlgEngine (RL ファインチューニング)
├── Dockerfile.simengine             # WorldEngine SimEngine (3DGS 閉ループシミュレーション)
├── docker-compose.yml               # 全サービス定義
├── docker-entrypoint-simengine.sh   # SimEngine Ray 起動スクリプト
├── Makefile                         # ビルド・運用コマンド集
├── .env.example                     # 環境変数テンプレート
└── README.md                        # 本ファイル
```

### イメージの継承関係

```
nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04   (DockerHub)
        │
        ▼
Dockerfile.base          → uniad-worldengine-base:latest
        │
        ├──▶ Dockerfile.uniad2       → uniad2:latest
        ├──▶ Dockerfile.algengine    → algengine:latest
        └──▶ Dockerfile.simengine    → simengine:latest
```

`Dockerfile.base` が APT パッケージ・Python 3.9・PyTorch 2.0.1+cu118 を共有するため、  
子イメージは各環境固有の依存のみを追加します。

---

## 前提条件

### ホスト要件

| 項目 | 要件 |
|------|------|
| OS | Ubuntu 22.04 LTS |
| GPU | NVIDIA A100 推奨 (最低 RTX 3090) |
| NVIDIA ドライバ | 525 以上 (CUDA 11.8 対応) |
| Docker | 24.0 以上 |
| Docker Compose | v2.20 以上 (`docker compose` コマンド) |
| NVIDIA Container Toolkit | インストール済み |
| ストレージ | 1 TB 以上の NVMe (推奨 4 TB) |
| RAM | 64 GB 以上 (推奨 256 GB) |

### NVIDIA Container Toolkit のインストール確認

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# → GPU 情報が表示されれば OK
```

インストールされていない場合:

```bash
# NVIDIA Container Toolkit インストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## クイックスタート

### 1. リポジトリのセットアップ

```bash
git clone <this-repo> uniad-worldengine-docker
cd uniad-worldengine-docker
```

### 2. 環境変数の設定

```bash
cp .env.example .env
```

`.env` を編集してデータパスと GPU 割り当てを設定します。

```bash
# .env の主要設定
HOST_DATA_ROOT=/mnt/data      # nuScenes / nuPlan 生データのパス
HOST_CKPT_DIR=/mnt/ckpts      # チェックポイントのパス
UNIAD_GPUS=0,1,2,3,4,5,6,7   # UniAD 学習に使用する GPU
ALGE_GPUS=0,1,2,3,4,5,6,7    # AlgEngine に使用する GPU
SIME_GPUS=4,5,6,7             # SimEngine に使用する GPU
```

### 3. イメージのビルド

```bash
# 全イメージを順番にビルド (最初の 1 回は 60〜90 分かかります)
make build-all
```

### 4. 動作確認

```bash
# 各コンテナが起動することを確認
make up-uniad2
docker exec -it uniad2 python3 -c "import torch; print(torch.cuda.is_available())"
# → True

make down
```

---

## Docker イメージのビルド手順

### ビルドコマンド一覧

```bash
make help   # コマンド一覧を表示
```

### ① ベースイメージ (必ず最初に実行)

```bash
make build-base
```

- 所要時間: 約 15〜20 分
- 内容: APT パッケージ、Python 3.9、PyTorch 2.0.1+cu118
- 生成イメージ: `uniad-worldengine-base:latest`

### ② 子イメージのビルド

`build-base` 完了後、3 イメージを目的に応じてビルドします。

**直列ビルド (安全・推奨)**

```bash
make build-all
# 内部で build-base → build-uniad2 → build-algengine → build-simengine の順に実行
```

**並列ビルド (高速・ディスク容量に余裕がある場合)**

```bash
make build-base           # ① 先に base を単独で完了させる
make build-parallel -j3   # ② ③ ④ を同時ビルド
```

**個別ビルド**

```bash
make build-uniad2     # UniAD 事前学習イメージのみ再ビルド
make build-algengine  # AlgEngine イメージのみ再ビルド
make build-simengine  # SimEngine イメージのみ再ビルド
```

> **注意**: 個別ビルド時も `build-base` が依存として自動実行されます。  
> base が既にビルド済みなら再ビルドは数秒で完了します。

### ビルド結果の確認

```bash
docker images | grep -E "uniad|algengine|simengine|base"
```

期待される出力:

```
uniad-worldengine-base   latest   xxxxxxxxxxxx   xx min ago   18.5GB
uniad2                   latest   xxxxxxxxxxxx   xx min ago   25.3GB
algengine                latest   xxxxxxxxxxxx   xx min ago   28.1GB
simengine                latest   xxxxxxxxxxxx   xx min ago   22.7GB
```

### 事前学習済み重みをビルド時に組み込む (オプション)

```bash
docker build \
    -f Dockerfile.uniad2 \
    --build-arg DOWNLOAD_CKPTS=1 \
    -t uniad2:latest \
    .
```

デフォルトは `DOWNLOAD_CKPTS=0` で重みは含まれません。  
ボリュームマウントで実行時に提供する方法を推奨します (後述)。

---

## コンテナの起動と操作

### サービス起動

```bash
# UniAD 事前学習コンテナ
make up-uniad2

# AlgEngine (RL ファインチューニング)
make up-algengine

# SimEngine (3DGS 閉ループ) — HEAD + WORKER を同時起動
make up-simengine

# 全サービスを一括起動
make up-all

# TensorBoard を追加起動 (port 6006)
make tensorboard
```

### コンテナの状態確認

```bash
docker compose ps
```

```
NAME               IMAGE              STATUS    PORTS
uniad2             uniad2:latest      running
algengine          algengine:latest   running
simengine-head     simengine:latest   running   0.0.0.0:8265->8265, 0.0.0.0:6379->6379
tensorboard        tensorflow:2.14.0  running   0.0.0.0:6006->6006
```

### コンテナへのシェルアクセス

```bash
# UniAD コンテナに入る
docker exec -it uniad2 /bin/bash

# AlgEngine コンテナに入る
docker exec -it algengine /bin/bash

# SimEngine HEAD に入る
docker exec -it simengine-head /bin/bash
```

### SimEngine Worker のスケールアップ

大規模なロールアウト生成時は Worker を追加します。

```bash
make scale-workers N=2   # Worker を 2 台に増やす
make scale-workers N=0   # Worker を停止
```

### 全コンテナの停止

```bash
make down
```

---

## 評価手順

### 評価フロー全体像

```
① nuScenes 標準評価 (uniad2)
        ↓
② Open-loop 評価 — NAVSIM navtest (algengine)
        ↓
③ 希少ケース抽出 (algengine)
        ↓
④ SimEngine ロールアウト生成 (simengine-head)
        ↓
⑤ Closed-loop 評価 (simengine-head)
        ↓
⑥ RL ファインチューニング (algengine)
        ↓
⑦ RL 後の Closed-loop 再評価 (simengine-head)
```

---

### ① nuScenes 標準評価

**使用コンテナ**: `uniad2`  
**評価内容**: Tracking / Mapping / Motion / Occupancy / Planning (nuScenes val セット)

```bash
# コンテナが起動していない場合は先に起動
make up-uniad2

# 評価実行 (8 GPU)
docker exec -it uniad2 bash -c "
    cd /workspace/UniAD
    ./tools/dist_test.sh \
        projects/configs/stage2_e2e/base_e2e.py \
        ckpts/uniad_base_e2e.pth \
        8 --eval bbox
"
```

期待される出力:

```
Tracking  AMOTA  : 0.380
Mapping   IoU    : 0.314
Motion    minADE : 0.794
Occupancy IoU-n  : 64.0
Planning  Col.   : 0.29%
```

**悪天候ロバスト性評価 (nuScenes-C)**

```bash
docker exec -it uniad2 bash -c "
    cd /workspace/UniAD
    for corruption in fog rain night motion_blur; do
        for severity in 1 2 3; do
            ./tools/dist_test.sh \
                projects/configs/stage2_e2e/base_e2e.py \
                ckpts/uniad_base_e2e.pth 4 --eval bbox \
                --cfg-options \
                data.test.ann_file=data/nuscenes_c/\${corruption}_s\${severity}_infos_val.pkl \
                2>&1 | tee logs/eval_\${corruption}_s\${severity}.log
        done
    done
"
```

---

### ② Open-loop 評価 (NAVSIM navtest)

**使用コンテナ**: `algengine`  
**評価内容**: PDMS (PDM Score) — navtest 全体 + 希少ケース分割

```bash
make up-algengine
make eval-openloop
```

内部では以下が実行されます:

```bash
# algengine コンテナ内
./scripts/e2e_dist_eval.sh \
    configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
    8
# → work_dirs/e2e_uniad_50pct_rlft_rare_log/navtest.csv に結果が保存される
```

**希少ケース専用評価**

```bash
docker exec -it algengine bash -c "
    cd \$ALGENGINE_ROOT
    ./scripts/e2e_dist_eval_navtest_failures.sh \
        configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
        work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
        8
"
# → navtest_failures.csv に保存
```

---

### ③ 希少ケース抽出

**使用コンテナ**: `algengine`  
**内容**: Open-loop 評価結果から失敗シナリオを 3 カテゴリで抽出

```bash
make extract-rare
```

抽出されるスプリットファイル:

```
configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare/
├── navtest_collision.yaml    ← 衝突あり (no_at_fault_collisions < 1.0)
├── navtest_off_road.yaml     ← 道路逸脱 (drivable_area_compliance < 1.0)
└── navtest_ep_1pct.yaml      ← 進行度下位 1% (ego_progress)
```

抽出されたシナリオ数を確認:

```bash
docker exec -it algengine bash -c "
    for f in configs/navsim_splits/navtest_split/e2e_uniad_50pct_rare/*.yaml; do
        echo \"\$f: \$(python3 -c \"import yaml; d=yaml.safe_load(open('\$f')); print(len(d.get('tokens',[])))\" ) scenarios\"
    done
"
```

---

### ④ SimEngine ロールアウト生成

**使用コンテナ**: `simengine-head`  
**内容**: 希少シナリオから閉ループロールアウトデータを生成

```bash
make up-simengine

# ロールアウト生成 (Non-Reactive モード)
make rollout
```

内部では以下が実行されます:

```bash
# simengine-head コンテナ内
bash scripts/run_ray_distributed_rollout.sh \
    /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct.py \
    /workspace/WorldEngine/data/alg_engine/ckpts/e2e_uniad_50pct_ep20.pth \
    e2e_uniad_50pct \
    navtrain_50pct_collision \
    navtrain
```

Ray Dashboard でロールアウトの進捗をリアルタイムで確認できます:

```
http://localhost:8265
```

生成されたロールアウトを AlgEngine 形式に変換:

```bash
docker exec -it simengine-head bash -c "
    cd \$SIMENGINE_ROOT
    python scripts/export_simulation_data.py \
        --test_path experiments/closed_loop_exps/e2e_uniad_50pct/navtrain_NR \
        --appendix \$(date +%Y%m%d)
"
# → /workspace/WorldEngine/data/alg_engine/openscene-synthetic/ に格納
```

**Reactive モードでのロールアウト** (他エージェントが反応する、より難しい設定)

```bash
docker exec -it simengine-head bash -c "
    cd \$SIMENGINE_ROOT
    bash scripts/run_ray_distributed_rollout.sh \
        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct.py \
        /workspace/WorldEngine/data/alg_engine/ckpts/e2e_uniad_50pct_ep20.pth \
        e2e_uniad_50pct \
        navtrain_50pct_collision \
        navtrain \
        R
"
```

---

### ⑤ Closed-loop 評価

**使用コンテナ**: `simengine-head`  
**内容**: 希少ケース 288 シナリオで閉ループ成功率・PDMS* を計測

```bash
# Non-Reactive (NR) モード
make eval-closedloop
```

内部では以下が実行されます:

```bash
# simengine-head コンテナ内
bash /workspace/WorldEngine/projects/AlgEngine/scripts/run_ray_distributed_testing.sh \
    /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    /workspace/WorldEngine/projects/AlgEngine/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
    e2e_uniad_rlft \
    navtest_failures \
    NR
```

**Reactive (R) モード** (より厳格な評価)

```bash
docker exec -it simengine-head bash -c "
    bash /workspace/WorldEngine/projects/AlgEngine/scripts/run_ray_distributed_testing.sh \
        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
        /workspace/WorldEngine/projects/AlgEngine/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
        e2e_uniad_rlft \
        navtest_failures \
        R
"
```

期待される結果 (WorldEngine フルパイプライン後):

```
Closed-loop Success Rate : 88.89%   (ベースモデル比 +15.28 pp)
Closed-loop PDMS*        : 70.12    (ベースモデル比 +9.84)
```

---

### ⑥ RL ファインチューニング

**使用コンテナ**: `algengine`  
**内容**: 希少ロールアウトを使った RL によるプランニング Head の強化

```bash
make up-algengine
make train-rl
```

内部では以下が実行されます:

```bash
# algengine コンテナ内
./scripts/e2e_dist_train.sh \
    configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
    8 \
    work_dirs/e2e_uniad_50pct/epoch_20.pth
# → work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_*.pth に保存
```

学習進捗は TensorBoard で確認:

```bash
make tensorboard
# http://localhost:6006 を開く
```

---

### ⑦ RL 後の再評価

RL ファインチューニング完了後に ⑤ Closed-loop 評価を再実行します。  
チェックポイントパスを新しいものに更新してから実行します:

```bash
docker exec -it simengine-head bash -c "
    bash /workspace/WorldEngine/projects/AlgEngine/scripts/run_ray_distributed_testing.sh \
        /workspace/WorldEngine/projects/AlgEngine/configs/worldengine/e2e_uniad_50pct_rlft_rare_log.py \
        /workspace/WorldEngine/projects/AlgEngine/work_dirs/e2e_uniad_50pct_rlft_rare_log/epoch_8.pth \
        e2e_uniad_rlft_final \
        navtest_failures \
        NR
"
```

---

## UniAD 事前学習手順

### Stage 1 — Perception 学習 (20 エポック)

```bash
make up-uniad2
make train-stage1
```

学習するモジュール: BEVFormer / TrackHead / MapHead  
所要時間の目安: A100×8 で約 3 日

### Stage 2 — End-to-End 学習 (6 エポック)

```bash
make train-stage2
```

学習するモジュール: 全 Head (+ MotionHead / OccHead / PlanningHead)  
所要時間の目安: A100×8 で約 1 日

**悪天候オーグメンテーション付き Stage 2**

```bash
docker exec -it uniad2 bash -c "
    cd /workspace/UniAD
    ./tools/dist_train.sh \
        projects/configs/stage2_e2e/base_e2e_adverse.py \
        8 \
        --work-dir work_dirs/stage2_e2e_adverse
"
```

**チェックポイントから再開**

```bash
docker exec -it uniad2 bash -c "
    cd /workspace/UniAD
    ./tools/dist_train.sh \
        projects/configs/stage2_e2e/base_e2e.py 8 \
        --resume-from work_dirs/stage2_e2e/epoch_3.pth
"
```

**nuScenes-mini での動作確認 (単一 GPU・2 分で完了)**

```bash
docker exec -it uniad2 bash -c "
    cd /workspace/UniAD
    ./tools/dist_train.sh \
        projects/configs/stage2_e2e/base_e2e.py 1 \
        --cfg-options \
            data.train.ann_file=data/infos/nuscenes_infos_temporal_train_mini.pkl \
            data.val.ann_file=data/infos/nuscenes_infos_temporal_val_mini.pkl \
            runner.max_epochs=2 \
            evaluation.interval=1
"
```

---

## ログとモニタリング

### TensorBoard

```bash
make tensorboard
# http://localhost:6006
```

### コンテナログ

```bash
# リアルタイムでログを追う
docker logs -f uniad2
docker logs -f algengine
docker logs -f simengine-head

# 直近 100 行
docker logs --tail=100 simengine-head
```

### GPU 使用率

```bash
# ホストから全 GPU を監視
watch -n 1 nvidia-smi

# コンテナ内から監視
docker exec -it algengine watch -n 1 nvidia-smi
```

### Ray Dashboard (SimEngine)

```
http://localhost:8265
```

ロールアウトの並列実行状況、GPU 使用率、失敗したタスクを確認できます。

---

## データボリューム管理

### ボリューム一覧

| Docker ボリューム | マウント先 (コンテナ内) | 用途 |
|-----------------|----------------------|------|
| `data_alg` | `/workspace/WorldEngine/data/alg_engine` | pkl / ckpts / 合成ロールアウト |
| `data_sim` | `/workspace/WorldEngine/data/sim_engine` | 3DGS シーン再構成 |
| `work_dirs` | `*/work_dirs` | 学習ログ・チェックポイント |
| `logs` | `*/logs` | TensorBoard ログ |

ホストの生データは `.env` の `HOST_DATA_ROOT` / `HOST_CKPT_DIR` で指定した  
ディレクトリから読み取り専用でマウントされます。

### チェックポイントのホスト側コピー

```bash
# algengine の work_dirs からホストへコピー
docker cp algengine:/workspace/WorldEngine/projects/AlgEngine/work_dirs/. \
    ./work_dirs_backup/
```

### ボリュームの確認

```bash
docker volume ls | grep uniad
docker volume inspect <volume_name>
```

---

## クリーンアップ

### コンテナのみ停止

```bash
make down
```

### コンテナ・イメージ・ボリュームを全削除

```bash
make clean
```

> **警告**: `make clean` は Docker ボリューム (学習ログ・チェックポイント含む) も削除します。  
> 重要なファイルは事前にホスト側にコピーしてください。

### ディスク使用量の確認

```bash
docker system df
```

---

## トラブルシューティング

### イメージビルドが失敗する

```bash
# キャッシュを無効にして再ビルド
docker build --no-cache -f Dockerfile.base -t uniad-worldengine-base:latest .
```

### `make build-uniad2` が「base イメージが見つからない」で失敗する

`build-base` が完了している必要があります。

```bash
docker images | grep uniad-worldengine-base
# 表示されなければ先に実行
make build-base
```

### コンテナ起動時に `CUDA not available`

NVIDIA Container Toolkit が正しく設定されているか確認します。

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

失敗する場合は Docker デーモンを再起動します:

```bash
sudo systemctl restart docker
```

### `shm_size` 不足エラー

`docker-compose.yml` の `shm_size` を増やします (デフォルト 32 GB):

```yaml
x-gpu-base: &gpu-base
  shm_size: "64g"   # 32g → 64g
```

### SimEngine の Ray がタイムアウト

```bash
# Ray クラスタをリセット
docker exec -it simengine-head bash -c "ray stop --force && ray start --head --num-gpus=4"
```

### `make eval-closedloop` で `Connection refused`

SimEngine HEAD が起動していない、または Ray が立ち上がっていない可能性があります。

```bash
# HEAD の状態確認
docker exec -it simengine-head bash -c "ray status"

# 起動していなければ再起動
make down
make up-simengine
# 30 秒待ってから再実行
sleep 30 && make eval-closedloop
```

---

## ベンチマーク期待値

### UniAD ベースモデル (nuScenes val)

| メトリクス | 値 |
|-----------|-----|
| Tracking AMOTA | 0.380 |
| Mapping IoU-lane | 0.314 |
| Motion minADE | 0.794 |
| Occupancy IoU-n | 64.0 |
| Planning avg. Col. | 0.29% |

### WorldEngine ポストトレーニング前後の比較

| 手法 | OL-PDMS (common) | OL-PDMS (rare) | CL 成功率 | CL-PDMS* |
|------|-----------------|----------------|----------|---------|
| ベースモデル | 85.62 | 47.15 | 73.61% | 60.28 |
| RL (希少ログ) | 89.29 | 62.56 | 74.31% | 62.55 |
| **WorldEngine (full)** | **88.95** | **59.83** | **88.89%** | **70.12** |

---

## 参考リンク

- [UniAD GitHub (v2.0)](https://github.com/OpenDriveLab/UniAD/tree/v2.0)
- [WorldEngine GitHub](https://github.com/OpenDriveLab/WorldEngine)
- [NAVSIM v1.1](https://github.com/autonomousvision/navsim)
- [nuScenes 公式](https://www.nuscenes.org/download)
- [ACDC Dataset](https://acdc.vision.ee.ethz.ch/)
- [セットアップガイド (日本語)](./uniad_worldengine_noconda_guide.md)
- [セットアップガイド (English)](./uniad_worldengine_noconda_guide_en.md)
