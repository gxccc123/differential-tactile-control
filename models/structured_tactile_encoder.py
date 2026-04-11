"""
StructuredTactileEncoder
========================
Encodes a tactile history [B, T, D] into two semantic tokens:

  state_token   t^s  : [B, H]  — stable contact state
  dynamic_token t^dyn: [B, H]  — velocity + residual dynamics

Pipeline
--------
1. TactileStem   phi(x_t): per-frame shared MLP  → h_t [B, T, H_stem]
2. Decompose in feature space:
     s_t = temporal mean of h_t           (smooth baseline)
     v_t = h_t - shift(h_t, 1)           (feature-space velocity)
     r_t = h_t - s_t                     (residual from baseline)
3. Aggregate:
     t^s   = state_head( h_t.mean(1) )            [B, H]
     t^v   = v_proj( v_t.flatten(1) )             [B, H]
     t^r   = r_proj( r_t.flatten(1) )             [B, H]
     t^dyn = dyn_fusion( cat([t^v, t^r]) )        [B, H]
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Lightweight tactile stem
# ---------------------------------------------------------------------------

class TactileStem(nn.Module):
    """
    Per-frame shared MLP with LayerNorm.

    Applies the same projection to every frame independently:
        phi: R^D  →  R^H_stem

    Weight sharing across T frames keeps the stem lightweight and
    ensures the temporal structure is learned only in the decomposition
    stage (v_t, r_t), not baked into the stem.
    """

    def __init__(self, tactile_dim: int, stem_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(tactile_dim, stem_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(stem_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            h: [B, T, H_stem]
        """
        B, T, D = x.shape
        h = self.net(x.reshape(B * T, D))   # [B*T, H_stem]
        return h.view(B, T, -1)             # [B, T, H_stem]


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class StructuredTactileEncoder(nn.Module):
    """
    Args:
        tactile_dim     D  : feature dim per timestep  (default 120)
        tac_history_cnt T  : number of history frames  (default 18)
        stem_dim   H_stem  : intermediate feature dim   (default 64)
        hidden_dim      H  : output token dim           (default 256)

    Inputs (forward):
        tactile: [B, T, D]  or  [B, T*D]   (both accepted)

    Outputs (forward):
        state_token   : [B, H]
        dynamic_token : [B, H]
    """

    def __init__(
        self,
        tactile_dim: int = 120,
        tac_history_cnt: int = 18,
        stem_dim: int = 64,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.tactile_dim = tactile_dim
        self.tac_history_cnt = tac_history_cnt
        self.stem_dim = stem_dim
        self.hidden_dim = hidden_dim

        flat_dim = tac_history_cnt * stem_dim   # T * H_stem = 18 * 64 = 1152

        # 1. Stem: per-frame shared MLP
        self.stem = TactileStem(tactile_dim, stem_dim)

        # 2. State token head
        #    pools h_t along time, then projects
        self.state_head = nn.Linear(stem_dim, hidden_dim)

        # 3. Dynamic token heads
        self.v_proj = nn.Linear(flat_dim, hidden_dim)   # velocity sequence → token
        self.r_proj = nn.Linear(flat_dim, hidden_dim)   # residual sequence → token
        self.dyn_fusion = nn.Linear(2 * hidden_dim, hidden_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_3d(self, tactile: torch.Tensor) -> torch.Tensor:
        """Accept [B, T, D] or [B, T*D]; always return [B, T, D]."""
        if tactile.dim() == 2:
            # flat input: [B, T*D]
            B = tactile.shape[0]
            return tactile.view(B, self.tac_history_cnt, self.tactile_dim)
        elif tactile.dim() == 3:
            return tactile
        else:
            raise ValueError(
                f"Expected tactile with 2 or 3 dims, got shape {tactile.shape}"
            )

    def _decompose(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose feature sequence h [B, T, H_stem] into s_t, v_t, r_t.

        s_t: temporal mean (smooth baseline)       [B, T, H_stem]
        v_t: feature-space velocity  d_0=0         [B, T, H_stem]
        r_t: residual from smooth baseline         [B, T, H_stem]
        """
        # s_t — temporal mean, broadcast back to [B, T, H_stem]
        s_t = h.mean(dim=1, keepdim=True).expand_as(h)

        # v_t — causal first-order difference in feature space
        v_t = torch.zeros_like(h)
        v_t[:, 1:, :] = h[:, 1:, :] - h[:, :-1, :]   # d_0 = 0

        # r_t — residual from smooth state
        r_t = h - s_t

        return s_t, v_t, r_t

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, tactile: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tactile: [B, T, D]  or  [B, T*D]

        Returns:
            state_token  : [B, H]   — t^s
            dynamic_token: [B, H]   — t^dyn
        """
        x = self._ensure_3d(tactile)         # [B, T, D]
        B, T, _ = x.shape

        # ── 1. Stem ──────────────────────────────────────────────────
        h = self.stem(x)                     # [B, T, H_stem]

        # ── 2. Decompose ─────────────────────────────────────────────
        s_t, v_t, r_t = self._decompose(h)  # each [B, T, H_stem]

        # ── 3. State token ───────────────────────────────────────────
        # Aggregate smooth state via temporal mean pooling
        s_agg = h.mean(dim=1)                # [B, H_stem]
        state_token = self.state_head(s_agg) # [B, H]

        # ── 4. Dynamic token ─────────────────────────────────────────
        t_v = self.v_proj(v_t.reshape(B, -1))                       # [B, H]
        t_r = self.r_proj(r_t.reshape(B, -1))                       # [B, H]
        dynamic_token = self.dyn_fusion(torch.cat([t_v, t_r], dim=-1))  # [B, H]

        return state_token, dynamic_token

    # ------------------------------------------------------------------
    # Debug helper: returns all intermediate tensors
    # ------------------------------------------------------------------

    def forward_debug(self, tactile: torch.Tensor) -> dict:
        """Same as forward but also returns intermediate tensors for inspection."""
        x = self._ensure_3d(tactile)
        B, T, _ = x.shape

        h = self.stem(x)
        s_t, v_t, r_t = self._decompose(h)

        s_agg = h.mean(dim=1)
        state_token = self.state_head(s_agg)

        t_v = self.v_proj(v_t.reshape(B, -1))
        t_r = self.r_proj(r_t.reshape(B, -1))
        dynamic_token = self.dyn_fusion(torch.cat([t_v, t_r], dim=-1))

        return {
            "x":             x,
            "h_t":           h,
            "s_t":           s_t,
            "v_t":           v_t,
            "r_t":           r_t,
            "s_agg":         s_agg,
            "t_v":           t_v,
            "t_r":           t_r,
            "state_token":   state_token,
            "dynamic_token": dynamic_token,
        }
