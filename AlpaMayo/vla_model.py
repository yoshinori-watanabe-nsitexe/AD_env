"""
vla_model.py
Alpamayo-R1 スタイルの VLA モック実装

実モデル (nvidia/Alpamayo-R1-10B) のアーキテクチャに基づき、
CPU 環境で動作する縮小版を提供する。
本番切り替え時は REAL_MODEL_ID を設定して load_real_model() を使う。

Alpamayo-R1 アーキテクチャ概要:
  - Vision Encoder : SigLIP-400M 相当 (ViT-L)
  - Language/Reasoning : Qwen2.5-7B 相当 LLM
  - Trajectory Decoder : Diffusion-based waypoint head
  - 入力 : 4-camera × 10Hz, 320×576 (downsampled from 1080×1920)
  - 出力 : chain-of-thought reasoning text + trajectory waypoints
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 設定クラス
# ---------------------------------------------------------------------------

@dataclass
class VisionConfig:
    image_size: int = 224          # モック用縮小 (本番: 576)
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 256         # 本番: 1024 (ViT-L)
    num_hidden_layers: int = 4     # 本番: 24
    num_attention_heads: int = 8
    intermediate_size: int = 512
    dropout: float = 0.0


@dataclass
class LLMConfig:
    vocab_size: int = 1024         # 本番: 151936 (Qwen2.5 tokenizer)
    hidden_size: int = 256         # 本番: 3584
    num_hidden_layers: int = 4     # 本番: 28
    num_attention_heads: int = 8
    intermediate_size: int = 512
    max_position_embeddings: int = 512
    dropout: float = 0.0


@dataclass
class TrajectoryConfig:
    """Diffusion-based trajectory decoder 設定"""
    num_waypoints: int = 10        # 予測ウェイポイント数 (x, y, heading)
    waypoint_dim: int = 3          # (dx, dy, dheading)
    hidden_size: int = 256
    num_diffusion_steps: int = 4   # 本番: ~100


@dataclass
class AlpamayoConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    num_cameras: int = 4           # front-wide, front-tele, cross-left, cross-right
    projection_dim: int = 256


# ---------------------------------------------------------------------------
# Vision Encoder (SigLIP-ViT 縮小版)
# ---------------------------------------------------------------------------

class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.qkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        # (B, heads, N, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class VisionEncoderLayer(nn.Module):
    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_size)
        self.attn = VisionAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size),
            nn.GELU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class VisionEncoder(nn.Module):
    """SigLIP-ViT 縮小版: 単一カメラ画像 → patch embeddings"""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            cfg.num_channels, cfg.hidden_size,
            kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, cfg.hidden_size)
        )
        self.num_patches = num_patches
        self.layers = nn.ModuleList(
            [VisionEncoderLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.ln = nn.LayerNorm(cfg.hidden_size)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            features: (B, num_patches+1, hidden_size)
        """
        B = pixel_values.shape[0]
        x = self.patch_embed(pixel_values)          # (B, hidden, H', W')
        x = x.flatten(2).transpose(1, 2)            # (B, N, hidden)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # (B, N+1, hidden)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)


# ---------------------------------------------------------------------------
# LLM (Qwen2/Llama スタイル 縮小版)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.q = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim
        q = self.q(x).view(B, L, H, D).transpose(1, 2)
        k = self.k(x).view(B, L, H, D).transpose(1, 2)
        v = self.v(x).view(B, L, H, D).transpose(1, 2)
        scale = D ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        # Causal mask
        causal = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.out(out)


class LLMLayer(nn.Module):
    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.hidden_size)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.RMSNorm(cfg.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class LLMBackbone(nn.Module):
    """Qwen2.5 スタイルの因果 LLM"""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [LLMLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.ln = nn.RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, L, vocab_size)
            hidden: (B, L, hidden_size) - last layer hidden states
        """
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        hidden = self.ln(x)
        logits = self.lm_head(hidden)
        return logits, hidden


# ---------------------------------------------------------------------------
# Trajectory Decoder (Diffusion-based waypoint head 縮小版)
# ---------------------------------------------------------------------------

class TrajectoryDecoder(nn.Module):
    """
    Diffusion-based trajectory decoder
    LLM の最終 hidden state を条件として waypoint sequence を生成する
    """

    def __init__(self, cfg: TrajectoryConfig):
        super().__init__()
        self.num_waypoints = cfg.num_waypoints
        self.waypoint_dim = cfg.waypoint_dim

        # noise → clean waypoints の denoising MLP (U-Net の簡略版)
        self.time_embed = nn.Sequential(
            nn.Linear(1, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        self.cond_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.net = nn.Sequential(
            nn.Linear(
                cfg.num_waypoints * cfg.waypoint_dim + cfg.hidden_size,
                cfg.hidden_size
            ),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.num_waypoints * cfg.waypoint_dim),
        )

    def forward(
        self,
        noisy_waypoints: torch.Tensor,
        t: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_waypoints : (B, num_waypoints, waypoint_dim)
            t               : (B,) diffusion timestep in [0,1]
            conditioning    : (B, hidden_size) LLM hidden state [CLS]
        Returns:
            pred_noise: (B, num_waypoints, waypoint_dim)
        """
        B = noisy_waypoints.shape[0]
        t_emb = self.time_embed(t.unsqueeze(-1).float())      # (B, hidden)
        cond = self.cond_proj(conditioning) + t_emb            # (B, hidden)
        flat = noisy_waypoints.reshape(B, -1)                  # (B, N*D)
        x = torch.cat([flat, cond], dim=-1)
        out = self.net(x)
        return out.reshape(B, self.num_waypoints, self.waypoint_dim)

    @torch.no_grad()
    def sample(
        self,
        conditioning: torch.Tensor,
        num_steps: int = 4,
    ) -> torch.Tensor:
        """DDIM-style simplified sampling"""
        B = conditioning.shape[0]
        wp = torch.randn(
            B, self.num_waypoints, self.waypoint_dim,
            device=conditioning.device
        )
        for i in reversed(range(num_steps)):
            t = torch.full((B,), i / num_steps, device=conditioning.device)
            noise_pred = self(wp, t, conditioning)
            # simplified DDIM step
            alpha = 1.0 - (i / num_steps)
            wp = (wp - (1 - alpha) ** 0.5 * noise_pred) / alpha ** 0.5
        return wp


# ---------------------------------------------------------------------------
# メイン VLA モデル
# ---------------------------------------------------------------------------

class AlpamayoVLA(nn.Module):
    """
    Alpamayo-R1 スタイルの VLA モデル

    入力:
      - multi_camera_images: (B, num_cameras, C, H, W)
      - input_ids          : (B, L) テキストトークン列
    出力:
      - logits             : (B, L, vocab_size) 次トークン予測
      - trajectory         : (B, num_waypoints, 3) 軌道予測
    """

    def __init__(self, cfg: AlpamayoConfig = AlpamayoConfig()):
        super().__init__()
        self.cfg = cfg

        # Vision Encoder (カメラ共有)
        self.vision_encoder = VisionEncoder(cfg.vision)

        # Vision → LLM 次元変換
        num_patches = (cfg.vision.image_size // cfg.vision.patch_size) ** 2 + 1
        self.vision_proj = nn.Linear(
            cfg.vision.hidden_size * cfg.num_cameras,
            cfg.llm.hidden_size
        )
        self.vision_tokens = num_patches  # カメラごとのトークン数

        # LLM Backbone
        self.llm = LLMBackbone(cfg.llm)

        # Trajectory Decoder
        self.trajectory_decoder = TrajectoryDecoder(cfg.trajectory)

        # LLM hidden → trajectory conditioning
        self.traj_cond_proj = nn.Linear(cfg.llm.hidden_size, cfg.trajectory.hidden_size)

    def encode_cameras(self, multi_camera_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            multi_camera_images: (B, num_cameras, C, H, W)
        Returns:
            vision_embeds: (B, num_patches, llm_hidden)
        """
        B, N, C, H, W = multi_camera_images.shape
        imgs = multi_camera_images.view(B * N, C, H, W)
        feats = self.vision_encoder(imgs)          # (B*N, patches, vis_hidden)
        _, P, D = feats.shape
        feats = feats.view(B, N, P, D)
        # カメラ次元を concat してから投影
        feats = feats.permute(0, 2, 1, 3).reshape(B, P, N * D)
        return self.vision_proj(feats)             # (B, P, llm_hidden)

    def forward(
        self,
        multi_camera_images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Returns dict with keys: logits, trajectory, vision_embeds
        """
        # 1. Vision encoding
        vision_embeds = self.encode_cameras(multi_camera_images)  # (B, P, H)

        # 2. Text embedding
        text_embeds = self.llm.embed(input_ids)                   # (B, L, H)

        # 3. Fusion: vision tokens を先頭に prepend
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)  # (B, P+L, H)

        # 4. LLM forward
        logits, hidden = self.llm(inputs_embeds=inputs_embeds)

        # 5. Trajectory: CLS 位置 (先頭) の hidden を使用
        traj_cond = self.traj_cond_proj(hidden[:, 0, :])          # (B, traj_hidden)
        B = multi_camera_images.shape[0]
        noisy_wp = torch.randn(
            B, self.cfg.trajectory.num_waypoints,
            self.cfg.trajectory.waypoint_dim,
            device=hidden.device
        )
        t = torch.zeros(B, device=hidden.device)
        trajectory = self.trajectory_decoder(noisy_wp, t, traj_cond)

        return {
            "logits": logits,
            "trajectory": trajectory,
            "vision_embeds": vision_embeds,
            "hidden": hidden,
        }

    @torch.no_grad()
    def generate_trajectory(self, multi_camera_images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """推論時: diffusion sampling で軌道を生成"""
        vision_embeds = self.encode_cameras(multi_camera_images)
        text_embeds = self.llm.embed(input_ids)
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        _, hidden = self.llm(inputs_embeds=inputs_embeds)
        traj_cond = self.traj_cond_proj(hidden[:, 0, :])
        return self.trajectory_decoder.sample(traj_cond)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_mock_model(scale: str = "base") -> AlpamayoVLA:
    """
    scale: "base" (デフォルト, ~5M params)
           "tiny" (~1M params, 高速テスト用)
    """
    if scale == "tiny":
        cfg = AlpamayoConfig(
            vision=VisionConfig(image_size=64, hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128),
            llm=LLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=128, vocab_size=512),
            trajectory=TrajectoryConfig(hidden_size=64),
        )
    else:
        cfg = AlpamayoConfig()
    return AlpamayoVLA(cfg)
