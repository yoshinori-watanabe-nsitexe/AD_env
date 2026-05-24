"""
vla_dataset.py
Alpamayo-R1 / AlpaSim 形式のテストデータセット

実データ形式:
  - AlpaSim: 標準化ベンチマーク (https://github.com/nvidia/alpasim)
  - nvidia/Alpamayo-R1-10B の HuggingFace model card に基づく入出力仕様
  - 入力: 4カメラ (front-wide, front-tele, cross-left, cross-right)
  - 出力: Chain-of-Thought テキスト + waypoint trajectory

本スクリプトでは実データの代わりにシナリオ定義から合成データを生成する。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# シナリオ定義 (AlpaSim ベンチマーク相当)
# ---------------------------------------------------------------------------

SCENARIO_TEMPLATES = [
    # (scenario_name, description, ideal_reasoning_keywords, expected_action)
    ("lane_keep",      "直線道路を一定速で走行",
     ["車線", "維持", "直進"],              "maintain_lane"),
    ("lane_change_left",  "左車線への車線変更",
     ["左", "車線変更", "ミラー確認"],       "change_left"),
    ("lane_change_right", "右車線への車線変更",
     ["右", "車線変更", "ウィンカー"],       "change_right"),
    ("intersection_straight", "信号のある交差点を直進",
     ["信号", "青", "直進"],               "go_straight"),
    ("intersection_left", "信号のある交差点を左折",
     ["左折", "歩行者", "確認"],            "turn_left"),
    ("intersection_right", "信号のある交差点を右折",
     ["右折", "対向", "待機"],              "turn_right"),
    ("pedestrian_crossing", "横断歩道に歩行者あり",
     ["歩行者", "一時停止", "待機"],         "stop"),
    ("vehicle_following", "前方車両に追従",
     ["前方", "車間距離", "速度調整"],       "follow"),
    ("emergency_stop", "突発的障害物で緊急停止",
     ["障害物", "ブレーキ", "停止"],         "emergency_stop"),
    ("roundabout", "ラウンドアバウト通行",
     ["優先", "合流", "ラウンドアバウト"],   "yield_then_merge"),
    ("highway_merge", "高速道路への合流",
     ["加速", "合流", "車間"],              "accelerate_merge"),
    ("parking", "駐車スペースへの駐車",
     ["駐車", "後退", "アライン"],          "park"),
]

ACTION_CLASSES = [s[3] for s in SCENARIO_TEMPLATES]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_CLASSES)}


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------

@dataclass
class VLASample:
    """1サンプル = 1推論ステップ"""
    scenario: str
    description: str
    # 画像: (num_cameras, C, H, W) - 実際は RGB だが合成なのでノイズ
    multi_camera_images: torch.Tensor
    # テキスト入力: instruction token ids
    input_ids: torch.Tensor
    # 正解軌道: (num_waypoints, 3) - (dx, dy, dheading) in vehicle frame
    gt_trajectory: torch.Tensor
    # 正解アクションクラス (整数)
    gt_action: int
    # Chain-of-Thought 参照テキスト (評価用)
    reference_reasoning: str
    # メタデータ
    sample_id: str = ""
    country: str = "JP"           # Alpamayo-R1 データは25カ国
    weather: str = "clear"
    time_of_day: str = "day"


def _make_gt_trajectory(action: str, num_waypoints: int = 10) -> torch.Tensor:
    """アクションに対応する合成参照軌道を生成"""
    wp = torch.zeros(num_waypoints, 3)  # (dx, dy, dheading)
    t = torch.linspace(0, 1, num_waypoints)

    if action == "maintain_lane":
        wp[:, 0] = t * 2.0          # 前進
    elif action == "change_left":
        wp[:, 0] = t * 2.0
        wp[:, 1] = -t * 1.5         # 左へ
        wp[:, 2] = torch.sin(t * 0.3)
    elif action == "change_right":
        wp[:, 0] = t * 2.0
        wp[:, 1] = t * 1.5          # 右へ
        wp[:, 2] = -torch.sin(t * 0.3)
    elif action in ("turn_left", "go_straight_left"):
        wp[:, 0] = torch.cos(t * math.pi / 2) * 3
        wp[:, 1] = -torch.sin(t * math.pi / 2) * 3
        wp[:, 2] = -t * (math.pi / 2)
    elif action in ("turn_right",):
        wp[:, 0] = torch.cos(t * math.pi / 2) * 3
        wp[:, 1] = torch.sin(t * math.pi / 2) * 3
        wp[:, 2] = t * (math.pi / 2)
    elif action in ("stop", "emergency_stop"):
        wp[:, 0] = t * 0.1          # ほぼ停止
    elif action == "follow":
        wp[:, 0] = t * 1.5          # 低速追従
    elif action == "accelerate_merge":
        wp[:, 0] = t * 3.0 + t**2  # 加速
        wp[:, 1] = t * 0.5
    elif action == "park":
        wp[:, 0] = t * 0.5
        wp[:, 1] = t * 0.8
        wp[:, 2] = t * (math.pi / 4)
    else:
        wp[:, 0] = t * 2.0
    return wp


import math


def _make_reference_reasoning(scenario: str, keywords: List[str]) -> str:
    """AlpaSim chain-of-thought 参照テキストを合成"""
    return (
        f"<think>\n"
        f"現在のシナリオ: {scenario}。\n"
        f"観察: " + "、".join(keywords) + "を確認。\n"
        f"判断: 安全マージンを確保しつつ適切な操作を実行する。\n"
        f"</think>\n"
        f"<action>{scenario}</action>"
    )


# ---------------------------------------------------------------------------
# データセットクラス
# ---------------------------------------------------------------------------

class AlpaSimDataset(Dataset):
    """
    AlpaSim ベンチマーク形式のモックデータセット

    実 Alpamayo-R1 の評価では:
      - 25カ国, 1,727時間のデータから抽出したシナリオを使用
      - closed-loop シミュレーション (AlpaSim フレームワーク)
      - メトリクス: offroad率, near-miss率, planning精度
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_cameras: int = 4,
        image_size: int = 224,
        num_waypoints: int = 10,
        seq_len: int = 32,
        vocab_size: int = 1024,
        seed: int = 42,
        split: str = "test",
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_cameras = num_cameras
        self.image_size = image_size
        self.num_waypoints = num_waypoints
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.split = split

        rng = random.Random(seed)
        torch.manual_seed(seed)

        self.samples: List[VLASample] = []
        countries = ["JP", "US", "DE", "FR", "CN", "KR", "GB", "AU", "BR", "IN"]
        weathers = ["clear", "rain", "fog", "snow", "night"]
        times = ["day", "dusk", "dawn", "night"]

        for i in range(num_samples):
            scenario_def = rng.choice(SCENARIO_TEMPLATES)
            name, desc, keywords, action = scenario_def

            # 合成画像 (正規化済み RGB)
            imgs = torch.randn(num_cameras, 3, image_size, image_size) * 0.5 + 0.5
            imgs = imgs.clamp(0, 1)

            # シナリオに対応したわずかな画像バイアスを追加 (視覚的手がかり)
            action_idx = ACTION_TO_IDX[action]
            imgs[:, 0, :20, :20] += action_idx * 0.01  # corner signal

            # テキストトークン列
            input_ids = torch.randint(0, vocab_size, (seq_len,))
            # BOS/EOS 相当
            input_ids[0] = 1   # BOS
            input_ids[-1] = 2  # EOS

            self.samples.append(VLASample(
                scenario=name,
                description=desc,
                multi_camera_images=imgs,
                input_ids=input_ids,
                gt_trajectory=_make_gt_trajectory(action, num_waypoints),
                gt_action=action_idx,
                reference_reasoning=_make_reference_reasoning(name, keywords),
                sample_id=f"{split}_{i:04d}",
                country=rng.choice(countries),
                weather=rng.choice(weathers),
                time_of_day=rng.choice(times),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VLASample:
        return self.samples[idx]

    def get_class_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.samples:
            act = ACTION_CLASSES[s.gt_action]
            counts[act] = counts.get(act, 0) + 1
        return dict(sorted(counts.items()))


def collate_fn(batch: List[VLASample]) -> Dict[str, torch.Tensor]:
    """DataLoader 用 collate"""
    return {
        "multi_camera_images": torch.stack([s.multi_camera_images for s in batch]),
        "input_ids": torch.stack([s.input_ids for s in batch]),
        "gt_trajectory": torch.stack([s.gt_trajectory for s in batch]),
        "gt_action": torch.tensor([s.gt_action for s in batch], dtype=torch.long),
        "scenario": [s.scenario for s in batch],
        "sample_id": [s.sample_id for s in batch],
    }
