"""Dense Information Guard (DIG) computation and dev-set quantile calibration (Sec. 3.2, Eq. 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np


def compute_frame_edge_and_text_stats(
    rgb_frame: np.ndarray,
    ocr_box_area_ratio: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute normalized OCR/stroke text-area proxy and Sobel/Laplacian high-frequency edge density.

    Args:
        rgb_frame: uint8 RGB array of shape (H, W, 3) or grayscale (H, W).
        ocr_box_area_ratio: Optional normalized bounding-box area from an OCR detector in [0, 1].
            When an external OCR detector is not invoked, uses connected high-contrast morphological
            stroke-region area ratio as a deterministic lightweight fallback.

    Returns:
        (area_ocr, edge_hf) both non-negative scalars.
    """
    if rgb_frame.ndim == 3:
        gray = (
            0.299 * rgb_frame[..., 0].astype(np.float32)
            + 0.587 * rgb_frame[..., 1].astype(np.float32)
            + 0.114 * rgb_frame[..., 2].astype(np.float32)
        ) / 255.0
    elif rgb_frame.ndim == 2:
        gray = rgb_frame.astype(np.float32) / (255.0 if rgb_frame.max() > 1.0 else 1.0)
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {rgb_frame.shape}")

    # Sobel horizontal/vertical gradients + discrete Laplacian via vectorized slicing
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    lap = np.zeros_like(gray)
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        gx[1:-1, 1:-1] = (
            (gray[:-2, 2:] + 2.0 * gray[1:-1, 2:] + gray[2:, 2:])
            - (gray[:-2, :-2] + 2.0 * gray[1:-1, :-2] + gray[2:, :-2])
        ) * 0.25
        gy[1:-1, 1:-1] = (
            (gray[2:, :-2] + 2.0 * gray[2:, 1:-1] + gray[2:, 2:])
            - (gray[:-2, :-2] + 2.0 * gray[:-2, 1:-1] + gray[:-2, 2:])
        ) * 0.25
        lap[1:-1, 1:-1] = np.abs(
            4.0 * gray[1:-1, 1:-1]
            - gray[:-2, 1:-1]
            - gray[2:, 1:-1]
            - gray[1:-1, :-2]
            - gray[1:-1, 2:]
        )

    sobel_mag = np.sqrt(gx * gx + gy * gy)
    edge_hf = float(0.5 * np.mean(sobel_mag) + 0.5 * np.mean(lap))

    if ocr_box_area_ratio is not None:
        area_ocr = float(np.clip(ocr_box_area_ratio, 0.0, 1.0))
    else:
        # High-contrast stroke density over 8x8 tiles (separates sharp text/charts from flat/natural scenes)
        h, w = gray.shape
        th, tw = max(1, h // 8), max(1, w // 8)
        cropped = gray[: th * 8, : tw * 8].reshape(th, 8, tw, 8)
        tile_contrast = cropped.max(axis=(1, 3)) - cropped.min(axis=(1, 3))
        tile_lap = lap[: th * 8, : tw * 8].reshape(th, 8, tw, 8).mean(axis=(1, 3))
        text_like_tiles = (tile_contrast > 0.35) & (tile_lap > 0.06)
        area_ocr = float(np.mean(text_like_tiles))

    return area_ocr, edge_hf


def _roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Exact Mann-Whitney U ROC-AUC without external sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    wins = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(wins / (len(pos) * len(neg)))


@dataclass
class DIGCalibrator:
    """Empirical CDF quantile map Calib(.) and tuned alpha_ocr for Eq. (2) (App. B.3)."""

    alpha_ocr: float = 0.65
    tau_dig: float = 0.70
    q_l1_dense_penalty: float = 0.35  # q_tier(c, R_L1) when phi_dig(c) > tau_dig
    sorted_dev_raw: Optional[np.ndarray] = None

    def raw_score(self, area_ocr: float, edge_hf: float) -> float:
        return float(self.alpha_ocr * area_ocr + (1.0 - self.alpha_ocr) * edge_hf)

    def fit_on_dev(
        self,
        dev_area_ocr: Sequence[float],
        dev_edge_hf: Sequence[float],
        dev_is_dense_labels: Optional[Sequence[int]] = None,
        dense_quantile_cutoff: float = 0.70,
    ) -> float:
        """Fit alpha_ocr via ROC-AUC (if labels provided) and freeze dev ECDF quantile map."""
        areas = np.asarray(dev_area_ocr, dtype=np.float64)
        edges = np.asarray(dev_edge_hf, dtype=np.float64)
        best_auc = 0.5
        if dev_is_dense_labels is not None and len(dev_is_dense_labels) == len(areas):
            y = np.asarray(dev_is_dense_labels, dtype=np.int32)
            if np.any(y == 1) and np.any(y == 0):
                best_alpha = self.alpha_ocr
                best_auc = -1.0
                best_sep = -1.0
                # Search alpha_ocr in [0.50, 0.95] so OCR text-area always anchors DIG
                # and break ROC-AUC ties by maximizing standardized separation margin (Cohen's d).
                for cand_alpha in np.linspace(0.50, 0.95, 19):
                    raw = cand_alpha * areas + (1.0 - cand_alpha) * edges
                    auc = _roc_auc_binary(y, raw)
                    pos_r = raw[y == 1]
                    neg_r = raw[y == 0]
                    pooled_sd = float(np.sqrt(0.5 * (np.var(pos_r) + np.var(neg_r)))) + 1e-8
                    sep = float((np.mean(pos_r) - np.mean(neg_r)) / pooled_sd)
                    if (auc > best_auc + 1e-9) or (abs(auc - best_auc) <= 1e-9 and sep > best_sep):
                        best_auc = auc
                        best_sep = sep
                        best_alpha = float(cand_alpha)
                self.alpha_ocr = best_alpha

        raw_dev = self.alpha_ocr * areas + (1.0 - self.alpha_ocr) * edges
        self.sorted_dev_raw = np.sort(raw_dev)
        self.tau_dig = float(np.clip(dense_quantile_cutoff, 0.05, 0.98))
        return best_auc

    def transform(self, area_ocr: float, edge_hf: float) -> float:
        """Map raw score onto [0, 1] via frozen development ECDF (Eq. 2)."""
        raw = self.raw_score(area_ocr, edge_hf)
        if self.sorted_dev_raw is None or len(self.sorted_dev_raw) == 0:
            return float(np.clip(raw, 0.0, 1.0))
        rank = int(np.searchsorted(self.sorted_dev_raw, raw, side="right"))
        return float(rank / len(self.sorted_dev_raw))

    def tier_fidelity(self, phi_dig: float, tier_is_l2: bool, disable_dig: bool = False) -> float:
        """Compute q_tier(c, tau) in (0, 1] (Sec. 4.3, Eq. 11).

        At R_L2, q_tier = 1.0. At R_L1 on dense clips (phi_dig > tau_dig), lowers score
        proportionally down to q_l1_dense_penalty while staying strictly > 0 so P_0 remains feasible.
        """
        if disable_dig:
            return 1.0 if tier_is_l2 else 0.85
        if tier_is_l2:
            return 1.0
        if phi_dig > self.tau_dig:
            excess = (phi_dig - self.tau_dig) / max(1e-6, 1.0 - self.tau_dig)
            val = 0.85 - excess * (0.85 - self.q_l1_dense_penalty)
            return float(np.clip(val, self.q_l1_dense_penalty, 1.0))
        return 0.90
