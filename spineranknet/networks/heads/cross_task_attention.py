"""Cross-task attention module for ranking score refinement.

After each per-task ranking head produces an independent "draft" score,
:class:`CrossTaskAttention` refines those scores by attending across all
T tasks simultaneously.  This lets the model learn clinical correlations
(e.g. severe Pfirrmann degeneration ↔ central canal stenosis) that
independent heads cannot capture.

A **zero-initialised residual gate** (``alpha``) ensures that the module
starts as an identity:  ``final = draft + alpha * delta``  with
``alpha = 0``.  This means warm-starting from a pre-trained checkpoint
produces identical outputs until the gate opens during training.

Modes
-----
``transformer``
    Self-attention over T task tokens (most expressive, ~67K params).
``mlp``
    Global T→T mapping with one hidden layer (~3.5K params).
``gated``
    Per-task gate from mean-pooled context (~0.4K params).
``mean_pool``
    Pull-toward-mean with no learnable parameters except ``alpha`` (1 param).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn


__all__ = ["CrossTaskAttention"]


class CrossTaskAttention(nn.Module):
    """Post-draft cross-task score refinement.

    Parameters
    ----------
    num_tasks : int
        Number of ranking tasks (T).
    d_model : int
        Internal token dimension (only used by ``transformer`` mode).
    nhead : int
        Number of attention heads (only used by ``transformer`` mode).
    num_layers : int
        Number of Transformer encoder layers.
    dim_ff : int
        Feed-forward hidden dimension inside Transformer layers.
    dropout : float
        Dropout probability.
    mode : str
        Refinement strategy: ``"transformer"`` | ``"mlp"`` | ``"gated"``
        | ``"mean_pool"``.
    store_attn : bool
        If True (and ``mode="transformer"``), capture attention weights
        in :attr:`last_attn_weights` for visualisation.
    """

    def __init__(
        self,
        num_tasks: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        mode: str = "transformer",
        store_attn: bool = False,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.mode = mode
        self.store_attn = store_attn

        # Zero-init residual gate: final = draft + alpha * delta
        self.alpha = nn.Parameter(torch.zeros(1))

        if mode == "transformer":
            # Project scalar score → d_model token
            self.score_proj = nn.Linear(1, d_model)
            # Learnable task-identity embeddings
            self.task_embeddings = nn.Parameter(
                torch.randn(num_tasks, d_model) * 0.02
            )
            # Transformer self-attention
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer, num_layers=num_layers,
            )
            # Project back to scalar delta
            self.output_proj = nn.Linear(d_model, 1)

            # Attention weight capture hooks
            self.last_attn_weights: Optional[torch.Tensor] = None
            if store_attn:
                self._register_attn_hooks()

        elif mode == "mlp":
            hidden = max(num_tasks * 2, 32)
            self.mlp = nn.Sequential(
                nn.Linear(num_tasks, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_tasks),
            )

        elif mode == "gated":
            # Per-task gate computed from all scores
            self.gate = nn.Linear(num_tasks, num_tasks)

        elif mode == "mean_pool":
            pass  # Only alpha (already created above)

        else:
            raise ValueError(
                f"Unknown cross-task mode: {mode!r}. "
                f"Choose from: transformer, mlp, gated, mean_pool"
            )

    # ------------------------------------------------------------------
    # Attention weight hooks (transformer mode only)
    # ------------------------------------------------------------------

    def _register_attn_hooks(self) -> None:
        """Wrap each layer's ``self_attn`` to capture attention weights."""
        # Set ``need_weights`` on every layer's self_attn and stash the
        # returned weights on the layer for later collection.
        for layer in self.transformer.layers:
            layer.self_attn._cta_store_attn = True
            original_forward = layer.self_attn.forward

            def _make_hook(
                orig_fn: Callable[..., tuple], layer_ref: nn.Module
            ) -> Callable[..., tuple]:
                def _hooked_forward(*a: object, **kw: object) -> tuple:
                    kw["need_weights"] = True
                    kw["average_attn_weights"] = False
                    out, weights = orig_fn(*a, **kw)
                    layer_ref._cta_last_weights = weights
                    return out, weights
                return _hooked_forward

            layer.self_attn.forward = _make_hook(
                original_forward, layer.self_attn
            )

    def _collect_attn_weights(self) -> Optional[torch.Tensor]:
        """Collect and stack attention weights from all layers.

        Returns
        -------
        torch.Tensor
            Shape ``(num_layers, B, nhead, T, T)``.
        """
        weights = []
        for layer in self.transformer.layers:
            w = getattr(layer.self_attn, "_cta_last_weights", None)
            if w is not None:
                weights.append(w)
        if weights:
            return torch.stack(weights, dim=0)  # (L, B, H, T, T)
        return None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, draft_scores: torch.Tensor) -> torch.Tensor:
        """Refine draft ranking scores via cross-task interaction.

        Parameters
        ----------
        draft_scores : torch.Tensor
            Shape ``(B, T)`` — draft scalar scores from per-task heads.

        Returns
        -------
        torch.Tensor
            Shape ``(B, T)`` — refined scores.
        """
        if self.mode == "transformer":
            # (B, T) → (B, T, 1) → (B, T, d_model)
            tokens = self.score_proj(draft_scores.unsqueeze(-1))
            # Add learnable task embeddings
            tokens = tokens + self.task_embeddings.unsqueeze(0)
            # Self-attention across tasks
            tokens = self.transformer(tokens)
            # Project back to scalar delta
            delta = self.output_proj(tokens).squeeze(-1)  # (B, T)

            # Capture attention weights for visualisation
            if self.store_attn:
                self.last_attn_weights = self._collect_attn_weights()

        elif self.mode == "mlp":
            delta = self.mlp(draft_scores)  # (B, T) → (B, T)

        elif self.mode == "gated":
            gate = torch.sigmoid(self.gate(draft_scores))  # (B, T)
            context = draft_scores.mean(dim=1, keepdim=True)  # (B, 1)
            delta = gate * (context - draft_scores)  # (B, T)

        elif self.mode == "mean_pool":
            context = draft_scores.mean(dim=1, keepdim=True)  # (B, 1)
            delta = context - draft_scores  # (B, T)

        # Zero-init residual gate
        return draft_scores + self.alpha * delta
