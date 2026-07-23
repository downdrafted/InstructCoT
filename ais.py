
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Paper Sec. 3.3: thresholds are calibrated once on a pool of this many samples.
DEFAULT_CALIBRATION_POOL_SIZE: int = 2000

# Paper Sec. 3.3: percentile ranks defining tau1 / tau2 (28% / 44% / 28% split).
DEFAULT_P_LOW: float = 28.0
DEFAULT_P_HIGH: float = 72.0

ScoreArray = Union[Sequence[float], np.ndarray, torch.Tensor]


def extract_first_answer_logits(
    logits_seq: torch.Tensor,
    answer_start_indices: torch.Tensor,
) -> torch.Tensor:
    if logits_seq.dim() != 3:
        raise ValueError(
            f"logits_seq must have shape (batch, seq_len, vocab); got {tuple(logits_seq.shape)}"
        )
    batch, seq_len, _ = logits_seq.shape
    if answer_start_indices.shape != (batch,):
        raise ValueError(
            f"answer_start_indices must have shape ({batch},); got {tuple(answer_start_indices.shape)}"
        )
    if bool((answer_start_indices < 0).any()) or bool((answer_start_indices >= seq_len).any()):
        raise ValueError("answer_start_indices out of range for seq_len")
    idx = answer_start_indices.view(batch, 1, 1).expand(-1, 1, logits_seq.size(-1))
    return logits_seq.gather(dim=1, index=idx).squeeze(1)


class AdaptiveInstructionSelector:
   

    def __init__(self) -> None:
        # Thresholds are undefined until `calibrate` is called on a pool of
        # entropy scores. They are then FIXED (paper: "Fixed once; NOT tuned
        # per dataset/model/test split").
        self.tau1: Optional[float] = None
        self.tau2: Optional[float] = None

    # ------------------------------------------------------------------
    # Complexity score:  s(v, q) = H( p_theta0(y | v, q) )  [bits]
    # ------------------------------------------------------------------
    @staticmethod
    def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
       
        if logits.dim() not in (1, 2):
            raise ValueError(
                f"logits must have shape (vocab,) or (batch, vocab); got {tuple(logits.shape)}"
            )
        log_p = F.log_softmax(logits.float(), dim=-1)          # nats
        p = log_p.exp()
        entropy_nats = -(p * log_p).sum(dim=-1)                # H in nats
        return entropy_nats / math.log(2.0)                    # -> bits (log base 2)

    # ------------------------------------------------------------------
    # Threshold calibration:  tau1 = P28, tau2 = P72 of the score pool
    # ------------------------------------------------------------------
    def calibrate(
        self,
        scores: ScoreArray,
        p_low: float = DEFAULT_P_LOW,
        p_high: float = DEFAULT_P_HIGH,
        force: bool = False,
    ) -> Tuple[float, float]:
        
        if (self.tau1 is not None or self.tau2 is not None) and not force:
            raise RuntimeError(
                "Thresholds already calibrated; the paper fixes them once. "
                "Pass force=True only for a deliberate re-calibration."
            )
        if not (0.0 <= p_low < p_high <= 100.0):
            raise ValueError(f"Require 0 <= p_low < p_high <= 100; got {p_low}, {p_high}")

        if isinstance(scores, torch.Tensor):
            pool = scores.detach().cpu().numpy().astype(np.float64).ravel()
        else:
            pool = np.asarray(scores, dtype=np.float64).ravel()
        if pool.size == 0:
            raise ValueError("Calibration pool is empty.")
        if not np.all(np.isfinite(pool)):
            raise ValueError("Calibration pool contains non-finite entropy scores.")

        # Paper equations: tau1 = Percentile_{28}(pool), tau2 = Percentile_{72}(pool)
        self.tau1 = float(np.percentile(pool, p_low))
        self.tau2 = float(np.percentile(pool, p_high))
        return self.tau1, self.tau2

    # ------------------------------------------------------------------
    # Routing rule:  s < tau1 -> 1;  tau1 <= s < tau2 -> 2;  s >= tau2 -> 3
    # ------------------------------------------------------------------
    def assign_level(self, score: float) -> int:
        if self.tau1 is None or self.tau2 is None:
            raise RuntimeError("Call calibrate() before assign_level().")
        s = float(score)
        if not math.isfinite(s):
            raise ValueError(f"Entropy score must be finite; got {s}")
        if s < self.tau1:
            return 1
        if s < self.tau2:  # tau1 <= s < tau2
            return 2
        return 3           # s >= tau2

    # ------------------------------------------------------------------
    # Instruction lookup for the routed level
    # ------------------------------------------------------------------
    @staticmethod
    def select_instruction(
        level: int,
        templates_by_level: Dict[int, Union[str, Sequence[str]]],
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        if level not in (1, 2, 3):
            raise ValueError(f"level must be in {{1, 2, 3}}; got {level}")
        if level not in templates_by_level:
            raise KeyError(f"No templates registered for level {level}")
        entry = templates_by_level[level]
        if isinstance(entry, str):
            return entry
        templates: List[str] = list(entry)
        if len(templates) == 0:
            raise ValueError(f"Empty template list for level {level}")
        if rng is None:
            return templates[0]
        return templates[int(rng.integers(len(templates)))]


# ----------------------------------------------------------------------
# Smoke test (synthetic logits only -- NOT experimental results)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np_rng = np.random.default_rng(0)

    vocab = 128
    pool_size = DEFAULT_CALIBRATION_POOL_SIZE  # 2000, per paper

    # Synthetic "first answer position" logits with varying sharpness to
    # produce a spread of entropies. Purely a functional smoke test.
    temps = torch.exp(torch.empty(pool_size).uniform_(math.log(0.05), math.log(5.0)))
    raw = torch.randn(pool_size, vocab)
    calib_logits = raw / temps.unsqueeze(1)

    selector = AdaptiveInstructionSelector()
    scores = selector.compute_entropy(calib_logits)
    assert scores.shape == (pool_size,)
    assert bool((scores >= 0).all()), "entropy must be non-negative"
    assert bool((scores <= math.log2(vocab) + 1e-4).all()), "entropy bounded by log2(vocab)"

    # Sanity: uniform logits -> log2(vocab) bits; near-one-hot -> ~0 bits.
    h_uniform = selector.compute_entropy(torch.zeros(vocab))
    assert abs(float(h_uniform) - math.log2(vocab)) < 1e-4
    one_hot = torch.full((vocab,), -50.0)
    one_hot[3] = 50.0
    assert float(selector.compute_entropy(one_hot)) < 1e-4

    tau1, tau2 = selector.calibrate(scores)  # 28th / 72nd percentiles
    assert tau1 < tau2

    levels = np.array([selector.assign_level(float(s)) for s in scores])
    frac = {lv: float((levels == lv).mean()) for lv in (1, 2, 3)}
    # By construction of the percentiles the split is ~28% / 44% / 28%.
    assert abs(frac[1] - 0.28) < 0.01
    assert abs(frac[2] - 0.44) < 0.01
    assert abs(frac[3] - 0.28) < 0.01

    templates: Dict[int, Union[str, Sequence[str]]] = {
        1: "Answer the question about the image. Output only the final answer.",
        2: ["Use relevant domain knowledge about the image's subject to answer."],
        3: "Reason step by step about the required capability, then answer.",
    }
    for lv in (1, 2, 3):
        instr = selector.select_instruction(lv, templates, rng=np_rng)
        assert isinstance(instr, str) and len(instr) > 0

    # "Fixed once" guard.
    try:
        selector.calibrate(scores)
        raise AssertionError("re-calibration without force=True should fail")
    except RuntimeError:
        pass

    print(f"[smoke] pool={pool_size} vocab={vocab}")
    print(f"[smoke] tau1(P28)={tau1:.4f} bits, tau2(P72)={tau2:.4f} bits (synthetic pool)")
    print(f"[smoke] level split: L1={frac[1]:.3f} L2={frac[2]:.3f} L3={frac[3]:.3f}")
    print("[smoke] all assertions passed")
