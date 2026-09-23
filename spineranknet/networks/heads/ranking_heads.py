"""Ranking head modules for scalar severity score prediction.

These heads map encoder features (or classification logits) to a single
scalar severity score per task.  They are used by
:class:`MultiHeadGradingRanker`.

All heads output scores on a **universal [0, UNIVERSAL_SCALE] scale**
(default 10.0) regardless of the task's number of classes.  This ensures
uniform gradient magnitudes, consistent loss hyperparameters, and
clinically interpretable severity scores.  The threshold-optimization
step maps these back to K task-specific classes at evaluation time.

Bounding modes
--------------
- ``"sigmoid"``  : ``sigmoid(x) * SCALE`` — classic, but saturates and
  kills gradients at the boundaries (especially problematic for small K).
- ``"softplus"`` (default) : Softplus-based soft clamping that provides
  non-zero gradients everywhere.  Inspired by the DeepRankSVM approach
  of using unbounded scores but adds gentle boundary guidance.
  ``softclamp(x) = SCALE * sp(x) / (sp(x) + sp(SCALE - x))``
  where ``sp = softplus``.  This converges to ``clamp(x, 0, SCALE)``
  for large beta but always has healthy gradients.
- ``"none"``     : Unbounded output — raw network output (as in the
  original DeepRankSVM paper).  The loss function alone constrains
  the score range.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from spineranknet.networks.heads.FasterKAN import FasterKAN

# ── Universal score range for all ranking heads ──────────────────────────
UNIVERSAL_SCALE: float = 10.0


def soft_clamp(x: torch.Tensor, lo: float = 0.0, hi: float = UNIVERSAL_SCALE,
               beta: float = 5.0) -> torch.Tensor:
    """Softplus-based soft clamping to [lo, hi] with non-zero gradients.

    Unlike sigmoid bounding, this function:
      - Has non-zero gradients everywhere (no saturation)
      - Is approximately linear in the [lo+eps, hi-eps] interior
      - Smoothly curves at the boundaries
      - Converges to hard clamp as beta → infinity

    Formula:
        ``soft_clamp(x) = lo + (hi - lo) * sp(x - lo) / (sp(x - lo) + sp(hi - x))``
        where ``sp(z) = softplus(beta * z) / beta``

    Parameters
    ----------
    x : Tensor
        Raw (unbounded) scores.
    lo, hi : float
        Lower and upper bounds.
    beta : float
        Sharpness of the clamping.  Higher values → closer to hard clamp.
        Default 5.0 gives a good balance of gradient flow and bounding.
    """
    # Shift to [0, range]
    rng = hi - lo
    z = x - lo

    sp_z = F.softplus(beta * z) / beta          # softplus(x - lo)
    sp_r = F.softplus(beta * (rng - z)) / beta   # softplus(hi - x)

    return lo + rng * sp_z / (sp_z + sp_r + 1e-8)


def _apply_bounding(score: torch.Tensor, bounded: bool,
                    bounding_mode: str = "softplus") -> torch.Tensor:
    """Apply score bounding to [0, UNIVERSAL_SCALE].

    Parameters
    ----------
    score : Tensor
        Raw unbounded scores.
    bounded : bool
        If False, return scores unchanged.
    bounding_mode : str
        ``"sigmoid"`` : sigmoid(x) * SCALE (legacy, gradient-killing)
        ``"softplus"`` : soft_clamp with healthy gradients (default)
        ``"none"`` : no bounding (unbounded, as in DeepRankSVM paper)
    """
    if not bounded or bounding_mode == "none":
        return score

    if bounding_mode == "sigmoid":
        return torch.sigmoid(score) * UNIVERSAL_SCALE

    # Default: softplus-based soft clamping
    return soft_clamp(score, lo=0.0, hi=UNIVERSAL_SCALE)


# ════════════════════════════════════════════════════════════════════════════
# INTERNAL RANKING NETS (architecture variants)
# ════════════════════════════════════════════════════════════════════════════

class _TransformerRankingNet(nn.Module):
    """CLS-token Transformer encoder → scalar."""

    def __init__(
        self,
        in_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        return self.head(x[:, 0]).squeeze(-1)


class _LeastSquaresRankingNet(nn.Module):
    """GELU-MLP optimised for L2 regression."""

    def __init__(
        self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _KANRankingNet(nn.Module):
    """FasterKAN-based ranking head — spline-based learnable activations."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_grids: int = 8,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.kan = FasterKAN(
            layers_hidden=[in_dim, hidden_dim, hidden_dim // 2],
            num_grids=num_grids,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.kan(x)
        x = self.dropout(x)
        return self.head(x).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# PUBLIC RANKING HEADS
# ════════════════════════════════════════════════════════════════════════════

class RankingHead(nn.Module):
    """Projects encoder features to a scalar severity score.

    Output is on the universal ``[0, UNIVERSAL_SCALE]`` scale (default 10).

    Supported head types:
      - ``"mlp"``           : Linear -> ReLU -> Dropout -> Linear
      - ``"linear"``        : Single linear projection
      - ``"transformer"``   : CLS-token Transformer encoder -> Linear
      - ``"least_squares"`` : GELU-MLP optimised for L2 regression
      - ``"kan"``           : FasterKAN with spline-based activations

    Bounding modes (``bounding_mode``):
      - ``"softplus"`` (default) : Gradient-friendly soft clamping
      - ``"sigmoid"``            : Classic sigmoid * SCALE (legacy)
      - ``"none"``               : Unbounded output
    """

    def __init__(self, in_dim: int, num_classes: int,
                 head_type: str = "mlp", hidden_dim: int = 128,
                 dropout: float = 0.3, bounded: bool = True,
                 bounding_mode: str = "softplus") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bounded = bounded
        self.head_type = head_type
        self.bounding_mode = bounding_mode

        if head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        elif head_type == "transformer":
            self.net = _TransformerRankingNet(
                in_dim, num_heads=4, num_layers=2,
                dim_ff=hidden_dim, dropout=dropout,
            )
        elif head_type == "least_squares":
            self.net = _LeastSquaresRankingNet(
                in_dim, hidden_dim=hidden_dim, dropout=dropout,
            )
        elif head_type == "kan":
            self.net = _KANRankingNet(
                in_dim, hidden_dim=hidden_dim, dropout=dropout,
            )
        else:  # linear
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type in ("transformer", "least_squares", "kan"):
            score = self.net(x)
        else:
            score = self.net(x).squeeze(-1)
        return _apply_bounding(score, self.bounded, self.bounding_mode)


class LogitRankingHead(nn.Module):
    """Derives a ranking score from classification logits via Linear(K, 1).

    Instead of learning to rank from raw encoder features, this head
    learns a linear combination of class logits -> scalar severity score.
    When ``bounded=True``, output is bounded to ``[0, UNIVERSAL_SCALE]``.

    Parameters
    ----------
    num_classes : int
        Number of classification output units (may be K or K-1 depending
        on the classification loss, e.g. CORN/CORAL/CLM use K-1).
    bounded : bool
        If True, apply bounding to ``[0, UNIVERSAL_SCALE]``.
    bounding_mode : str
        ``"softplus"`` (default), ``"sigmoid"``, or ``"none"``.
    """

    def __init__(self, num_classes: int, bounded: bool = True,
                 bounding_mode: str = "softplus") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bounded = bounded
        self.bounding_mode = bounding_mode
        self.linear = nn.Linear(num_classes, 1, bias=True)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, K) -> score: (B,)"""
        score = self.linear(logits).squeeze(-1)
        return _apply_bounding(score, self.bounded, self.bounding_mode)


class MLPLogitRankingHead(nn.Module):
    """Derives a ranking score from classification logits via a small MLP.

    ``logits (B, K) -> Linear(K, H) -> ReLU -> Dropout -> Linear(H, 1) -> score``

    More expressive than :class:`LogitRankingHead` (linear) while still
    operating on the task's own logits so that ranking gradients flow back
    through the classification head.

    Parameters
    ----------
    num_classes : int
        Number of classification output units (K or K-1).
    hidden_dim : int
        Hidden layer width.
    dropout : float
        Dropout probability between layers.
    bounded : bool
        If True, apply bounding to ``[0, UNIVERSAL_SCALE]``.
    bounding_mode : str
        ``"softplus"`` (default), ``"sigmoid"``, or ``"none"``.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 32,
                 dropout: float = 0.3, bounded: bool = True,
                 bounding_mode: str = "softplus") -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bounded = bounded
        self.bounding_mode = bounding_mode
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, K) -> score: (B,)"""
        score = self.net(logits).squeeze(-1)
        return _apply_bounding(score, self.bounded, self.bounding_mode)


class GELULogitRankingHead(nn.Module):
    """Derives a ranking score from logits via a GELU-MLP (least-squares style).

    ``logits (B, K) -> Linear(K, H) -> GELU -> Dropout -> Linear(H, H//2)
    -> GELU -> Linear(H//2, 1) -> score``

    Same architecture as :class:`_LeastSquaresRankingNet` but operating on
    K-dimensional logits instead of 512-dim encoder features.

    Parameters
    ----------
    num_classes : int
        Number of classification output units (K or K-1).
    hidden_dim : int
        Hidden layer width (default 32, smaller than feature-based since K is small).
    dropout : float
        Dropout probability between layers.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 32,
                 dropout: float = 0.3, bounded: bool = True,
                 bounding_mode: str = "softplus") -> None:
        super().__init__()
        self.bounded = bounded
        self.bounding_mode = bounding_mode
        self.net = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 4), 1),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        score = self.net(logits).squeeze(-1)
        return _apply_bounding(score, self.bounded, self.bounding_mode)


class KANLogitRankingHead(nn.Module):
    """Derives a ranking score from logits via FasterKAN.

    ``logits (B, K) -> FasterKAN([K, H, H//2]) -> Dropout -> Linear(H//2, 1) -> score``

    Parameters
    ----------
    num_classes : int
        Number of classification output units (K or K-1).
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 32,
                 dropout: float = 0.3, bounded: bool = True,
                 bounding_mode: str = "softplus") -> None:
        super().__init__()
        self.bounded = bounded
        self.bounding_mode = bounding_mode
        self.kan = FasterKAN(
            layers_hidden=[num_classes, hidden_dim, max(hidden_dim // 2, 4)],
            num_grids=8,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(max(hidden_dim // 2, 4), 1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        x = self.kan(logits)
        x = self.dropout(x)
        score = self.head(x).squeeze(-1)
        return _apply_bounding(score, self.bounded, self.bounding_mode)


class TransformerLogitRankingHead(nn.Module):
    """Derives a ranking score from classification logits via a Transformer.

    ``logits (B, K) -> unsqueeze -> CLS token + self-attention -> Linear(d, 1) -> score``

    Uses a CLS-token Transformer encoder that treats each logit dimension as
    a sequence element.  The CLS token attends to all K logit dimensions and
    its output is linearly projected to a scalar score.  This allows the head
    to learn non-linear, attention-weighted combinations of class logits.

    Since K is small (typically 2–5), the Transformer is lightweight:
    1 layer, 2 heads, small feedforward dim.

    Parameters
    ----------
    num_classes : int
        Number of classification output units (K or K-1).
    hidden_dim : int
        Transformer model dimension (default 32).
    num_heads : int
        Number of attention heads (default 2).
    num_layers : int
        Number of Transformer encoder layers (default 1).
    dropout : float
        Dropout probability (default 0.3).
    bounded : bool
        If True, apply bounding to ``[0, UNIVERSAL_SCALE]``.
    bounding_mode : str
        ``"softplus"`` (default), ``"sigmoid"``, or ``"none"``.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 32,
                 num_heads: int = 2, num_layers: int = 1,
                 dropout: float = 0.3, bounded: bool = True,
                 bounding_mode: str = "softplus") -> None:
        super().__init__()
        self.bounded = bounded
        self.bounding_mode = bounding_mode
        # Project each logit scalar to hidden_dim via a shared linear
        self.input_proj = nn.Linear(1, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        # Positional encoding for K + 1 tokens (CLS + K logits)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_classes + 1, hidden_dim) * 0.02
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, K) -> score: (B,)"""
        B, K = logits.shape
        # Treat each logit as a token: (B, K) -> (B, K, 1) -> (B, K, D)
        x = self.input_proj(logits.unsqueeze(-1))  # (B, K, D)
        cls = self.cls_token.expand(B, -1, -1)      # (B, 1, D)
        x = torch.cat([cls, x], dim=1)              # (B, K+1, D)
        x = x + self.pos_embed[:, :K + 1, :]        # add positional encoding
        x = self.transformer(x)                      # (B, K+1, D)
        score = self.head(x[:, 0]).squeeze(-1)       # CLS token -> scalar
        return _apply_bounding(score, self.bounded, self.bounding_mode)


class ExpectationRankingHead(nn.Module):
    """Derives a ranking score as the softmax-weighted expected grade.

    ``score = sum(softmax(logits) * linspace(0, UNIVERSAL_SCALE, K))``

    No learnable parameters -- ranking quality depends entirely on how
    well-calibrated the classification logits are.  This makes it a
    useful baseline: if classification is good, ranking is free.

    Scores are on the universal ``[0, UNIVERSAL_SCALE]`` scale.

    Parameters
    ----------
    num_classes : int
        Number of ordinal severity classes (K).  Must be the *true*
        class count, not K-1.  For CORN/CORAL losses (K-1 logits),
        cumulative probabilities are converted to class probabilities
        before computing the expectation.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer(
            "grade_values",
            torch.linspace(0.0, UNIVERSAL_SCALE, num_classes),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (B, K) or (B, K-1) -> score: (B,)

        When input has K-1 columns (ordinal cumulative logits), they are
        converted to K class probabilities via adjacent differences of
        cumulative sigmoid.
        """
        if logits.shape[1] == self.num_classes - 1:
            # Ordinal cumulative logits -> class probabilities
            cum = torch.sigmoid(logits)                    # (B, K-1)
            ones = torch.ones(
                logits.shape[0], 1, device=logits.device)
            zeros = torch.zeros(
                logits.shape[0], 1, device=logits.device)
            cum_ext = torch.cat([ones, cum, zeros], dim=1)  # (B, K+1)
            probs = cum_ext[:, :-1] - cum_ext[:, 1:]        # (B, K)
            probs = probs.clamp(min=0)
        else:
            probs = torch.softmax(logits, dim=1)            # (B, K)
        return (probs * self.grade_values).sum(dim=1)


# ════════════════════════════════════════════════════════════════════════════
# CONCATENATED-LOGIT RANKING HEADS  (inter-task modelling)
# ════════════════════════════════════════════════════════════════════════════

class ConcatLogitRankingHead(nn.Module):
    """Ranking head that operates on ALL task logits concatenated.

    Instead of per-task independent ranking, this head concatenates all
    classification logits ``(B, sum_K)`` into a single vector and uses a
    shared trunk to model inter-task relationships before producing
    per-task severity scores.

    This allows the model to learn correlations such as:
      - Severe Pfirrmann ↔ Central Canal Stenosis
      - Disc herniation ↔ Foraminal stenosis
      - Endplate defect ↔ Modic change

    Architecture variants (``head_type``):
      - ``"mlp"``: Shared MLP trunk → per-task linear output
      - ``"transformer"``: CLS tokens per task + self-attention → per-task score
      - ``"kan"``: Shared FasterKAN trunk → per-task linear output

    Parameters
    ----------
    task_dims : dict[str, int]
        Mapping ``{task_name: K}`` giving the logit dimension for each task.
        Only ranking tasks should be included (not IVDlevel).
    head_type : str
        Architecture: ``"mlp"`` | ``"transformer"`` | ``"kan"``.
    hidden_dim : int
        Hidden dimension of the shared trunk.
    dropout : float
        Dropout probability.
    bounded : bool
        Apply bounding to ``[0, UNIVERSAL_SCALE]``.
    bounding_mode : str
        Bounding function.
    """

    def __init__(
        self,
        task_dims: Dict[str, int],
        head_type: str = "mlp",
        hidden_dim: int = 64,
        dropout: float = 0.3,
        bounded: bool = True,
        bounding_mode: str = "softplus",
    ) -> None:
        super().__init__()
        self.task_names = list(task_dims.keys())
        self.task_dims = task_dims
        self.total_dim = sum(task_dims.values())
        self.bounded = bounded
        self.bounding_mode = bounding_mode
        self.head_type = head_type
        n_tasks = len(self.task_names)

        if head_type == "transformer":
            # Per-task CLS tokens + self-attention
            d_model = hidden_dim
            self.input_proj = nn.Linear(self.total_dim, d_model)
            self.task_tokens = nn.Parameter(
                torch.randn(1, n_tasks, d_model) * 0.02
            )
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=min(4, d_model),
                dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.output_heads = nn.ModuleDict({
                t: nn.Linear(d_model, 1) for t in self.task_names
            })
        elif head_type == "kan":
            self.trunk = FasterKAN(
                layers_hidden=[self.total_dim, hidden_dim, hidden_dim // 2],
                num_grids=8,
            )
            self.dropout = nn.Dropout(dropout)
            self.output_heads = nn.ModuleDict({
                t: nn.Linear(hidden_dim // 2, 1) for t in self.task_names
            })
        else:  # mlp (default)
            self.trunk = nn.Sequential(
                nn.Linear(self.total_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
            )
            self.output_heads = nn.ModuleDict({
                t: nn.Linear(hidden_dim // 2, 1) for t in self.task_names
            })

    def forward(self, concat_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        concat_logits : Tensor (B, sum_K)
            All task logits concatenated in the order of ``self.task_names``.

        Returns
        -------
        dict[str, Tensor]
            Per-task severity scores ``{task_name: (B,)}``.
        """
        if self.head_type == "transformer":
            B = concat_logits.shape[0]
            ctx = self.input_proj(concat_logits).unsqueeze(1)  # (B, 1, D)
            tokens = self.task_tokens.expand(B, -1, -1)         # (B, T, D)
            x = torch.cat([ctx, tokens], dim=1)                 # (B, 1+T, D)
            x = self.transformer(x)
            # Skip context token, use task tokens
            task_feats = x[:, 1:, :]  # (B, T, D)
            out = {}
            for i, t in enumerate(self.task_names):
                score = self.output_heads[t](task_feats[:, i]).squeeze(-1)
                out[t] = _apply_bounding(score, self.bounded, self.bounding_mode)
            return out
        elif self.head_type == "kan":
            h = self.trunk(concat_logits)
            h = self.dropout(h)
        else:
            h = self.trunk(concat_logits)

        out = {}
        for t in self.task_names:
            score = self.output_heads[t](h).squeeze(-1)
            out[t] = _apply_bounding(score, self.bounded, self.bounding_mode)
        return out


# ════════════════════════════════════════════════════════════════════════════
# FACTORY: map head_type string → logit-based ranking head class
# ════════════════════════════════════════════════════════════════════════════

LOGIT_HEAD_REGISTRY: Dict[str, type[nn.Module]] = {
    "linear":        LogitRankingHead,
    "mlp":           MLPLogitRankingHead,
    "transformer":   TransformerLogitRankingHead,
    "least_squares": GELULogitRankingHead,
    "kan":           KANLogitRankingHead,
    "expectation":   ExpectationRankingHead,
}


def build_logit_ranking_head(
    head_type: str,
    num_classes: int,
    bounded: bool = True,
    bounding_mode: str = "softplus",
    hidden_dim: int = 32,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a logit-based ranking head by type string.

    Parameters
    ----------
    head_type : str
        One of: ``"linear"``, ``"mlp"``, ``"transformer"``,
        ``"least_squares"``, ``"kan"``, ``"expectation"``.
    num_classes : int
        Number of classification logit outputs (K or K-1).

    Returns
    -------
    nn.Module
        Ranking head that maps ``(B, K) -> (B,)`` severity scores.
    """
    if head_type == "expectation":
        return ExpectationRankingHead(num_classes)
    if head_type == "linear":
        return LogitRankingHead(num_classes, bounded=bounded,
                                bounding_mode=bounding_mode)
    if head_type == "mlp":
        return MLPLogitRankingHead(num_classes, hidden_dim=hidden_dim,
                                   dropout=dropout, bounded=bounded,
                                   bounding_mode=bounding_mode)
    if head_type == "least_squares":
        return GELULogitRankingHead(num_classes, hidden_dim=hidden_dim,
                                    dropout=dropout, bounded=bounded,
                                    bounding_mode=bounding_mode)
    if head_type == "transformer":
        return TransformerLogitRankingHead(num_classes, hidden_dim=hidden_dim,
                                           dropout=dropout, bounded=bounded,
                                           bounding_mode=bounding_mode)
    if head_type == "kan":
        return KANLogitRankingHead(num_classes, hidden_dim=hidden_dim,
                                   dropout=dropout, bounded=bounded,
                                   bounding_mode=bounding_mode)
    raise ValueError(f"Unknown logit ranking head type: {head_type!r}. "
                     f"Available: {list(LOGIT_HEAD_REGISTRY.keys())}")
