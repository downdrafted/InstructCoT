from __future__ import annotations

from typing import Dict, Sequence, Union

import torch

Scalar = Union[torch.Tensor, float]

# --- Curriculum constants (Section 3.5) -------------------------------------
PHASE1_END: float = 0.15   # Phase 1 = first 15% of training
PHASE2_END: float = 0.40   # Phase 2 = next 25%  (0.15 + 0.25)
TRANSITION_BAND: float = 0.05  # 5% linear transition band at each boundary

# --- Loss coefficients (Section 3.5, composite-loss equation) ---------------
ALPHA_K: float = 0.3   # weight of each CoT-step CE term  L_CE(r_k)
BETA: float = 1.0      # weight of the final-answer CE term L_CE(y)
LAMBDA: float = 0.1    # weight of the ICA auxiliary term  L_ICA


def _as_tensor(x: Scalar) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(float(x))


class ProgressiveCoTCurriculum:
    MODES = ("direct", "rationale", "full_cot")

    def __init__(
        self,
        phase1_end: float = PHASE1_END,
        phase2_end: float = PHASE2_END,
        band: float = TRANSITION_BAND,
    ) -> None:
        if not (0.0 < phase1_end < phase2_end < 1.0):
            raise ValueError("Require 0 < phase1_end < phase2_end < 1.")
        if band < 0.0 or (phase1_end + band / 2.0) > (phase2_end - band / 2.0):
            raise ValueError("Transition bands must not overlap.")
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.band = band

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _clamp01(p: float) -> float:
        return min(1.0, max(0.0, float(p)))

    def _ramp(self, p: float, boundary: float) -> float:
        if self.band == 0.0:  # degenerate: hard switch at the boundary
            return 1.0 if p >= boundary else 0.0
        t = (p - boundary) / self.band
        return min(1.0, max(0.0, t))

    # -- public API ----------------------------------------------------------

    def phase(self, progress: float) -> int:
        p = self._clamp01(progress)
        if p < self.phase1_end:
            return 1
        if p < self.phase2_end:
            return 2
        return 3

    def schedule(self, progress: float) -> Dict[str, float]:
        p = self._clamp01(progress)
        t1 = self._ramp(p, self.phase1_end)  # direct   -> rationale blend
        t2 = self._ramp(p, self.phase2_end)  # rationale -> full_cot blend
        # Bands are guaranteed non-overlapping (checked in __init__), so:
        weights = {
            "direct": 1.0 - t1,
            "rationale": t1 - t2,
            "full_cot": t2,
        }
        return weights


def instructcot_loss(
    step_ce_losses: Sequence[Scalar],
    answer_ce: Scalar,
    ica_ce: Scalar,
    alpha: float = ALPHA_K,
    beta: float = BETA,
    lam: float = LAMBDA,
) -> torch.Tensor:
    answer_t = _as_tensor(answer_ce)
    ica_t = _as_tensor(ica_ce)
    loss = beta * answer_t + lam * ica_t
    for step_ce in step_ce_losses:
        loss = loss + alpha * _as_tensor(step_ce)
    return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    curriculum = ProgressiveCoTCurriculum()

    print("PCC curriculum mixing weights (band after each boundary; phases stay pure):")
    for p in (0.0, 0.15, 0.20, 0.40, 1.0):
        w = curriculum.schedule(p)
        print(
            f"  p={p:4.2f}  phase={curriculum.phase(p)}  "
            f"direct={w['direct']:.3f}  rationale={w['rationale']:.3f}  "
            f"full_cot={w['full_cot']:.3f}  (sum={sum(w.values()):.3f})"
        )

    # Loss example: 3 CoT steps (grounding, reasoning, synthesis).
    step_losses = [torch.tensor(2.0), torch.tensor(1.5), torch.tensor(1.0)]
    answer_ce = torch.tensor(1.2)
    ica_ce = torch.tensor(0.8)
    loss = instructcot_loss(step_losses, answer_ce, ica_ce)
    expected = 0.3 * (2.0 + 1.5 + 1.0) + 1.0 * 1.2 + 0.1 * 0.8
    print(
        f"\nComposite loss example: L = 0.3*(2.0+1.5+1.0) + 1.0*1.2 + 0.1*0.8"
        f" = {loss.item():.4f} (expected {expected:.4f})"
    )
    assert abs(loss.item() - expected) < 1e-6, "loss mismatch"

    # Gradient sanity check: loss is differentiable w.r.t. tensor inputs.
    a = torch.tensor(1.2, requires_grad=True)
    instructcot_loss([torch.tensor(2.0)], a, torch.tensor(0.8)).backward()
    assert a.grad is not None and abs(a.grad.item() - 1.0) < 1e-6
    print("Gradient check passed (dL/d answer_ce = beta = 1.0).")
