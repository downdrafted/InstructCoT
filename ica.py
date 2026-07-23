
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class InstructionConditionedAttention(nn.Module):
    #: Paper Sec. 3.4: L_I ~= 32 instruction tokens, hard cap 64.
    MAX_INSTRUCTION_LEN: int = 64

    def __init__(self, d: int, n_heads: int = 8, d_k: int = 64) -> None:
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_k = d_k
        self.inner_dim = n_heads * d_k  # 8 * 64 = 512

        # --- Eq. (ICA) projections: W_Q, W_K, W_V map d -> n_heads*d_k. ---
        # bias=False: the paper writes pure matrix products (F_V W_Q, ...).
        self.W_Q = nn.Linear(d, self.inner_dim, bias=False)
        self.W_K = nn.Linear(d, self.inner_dim, bias=False)
        self.W_V = nn.Linear(d, self.inner_dim, bias=False)
        # Output projection back to model width d (paper Sec. 3.4).
        self.W_O = nn.Linear(self.inner_dim, d, bias=False)

        # Xavier-uniform init for W_Q, W_K, W_V and the output projection.
        for proj in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.xavier_uniform_(proj.weight)

        # --- Gate: Flamingo-style tanh gate, learnable scalar g init 0. ---
        # gate = tanh(g); g = 0 at init  =>  gate = 0  =>  F'_V == F_V
        # exactly at step 0 (identity-at-init, "following Flamingo").
        # NOTE: the paper's sigma(g) notation is NOT used because
        # sigma(0) = 0.5 would violate identity-at-init; see module
        # docstring, item 4.
        self.g = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        F_V: torch.Tensor,
        E_I: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if F_V.dim() != 3 or E_I.dim() != 3:
            raise ValueError(
                f"F_V and E_I must be 3-D (B, seq, d); got {tuple(F_V.shape)} "
                f"and {tuple(E_I.shape)}"
            )
        B, N_V, d = F_V.shape
        B_i, L_I, d_i = E_I.shape
        if d != self.d or d_i != self.d:
            raise ValueError(
                f"Last dim of F_V ({d}) and E_I ({d_i}) must equal d={self.d}"
            )
        if B_i != B:
            raise ValueError(f"Batch mismatch: F_V has B={B}, E_I has B={B_i}")
        if L_I > self.MAX_INSTRUCTION_LEN:
            raise ValueError(
                f"L_I={L_I} exceeds the paper's instruction-length cap of "
                f"{self.MAX_INSTRUCTION_LEN} (Sec. 3.4: L_I ~= 32, cap 64)"
            )
        if key_padding_mask is not None and key_padding_mask.shape != (B, L_I):
            raise ValueError(
                f"key_padding_mask must have shape {(B, L_I)}; got "
                f"{tuple(key_padding_mask.shape)}"
            )

        # --- Projections (Eq. ICA): split into heads. -------------------
        # (B, seq, inner_dim) -> (B, n_heads, seq, d_k)
        Q = self.W_Q(F_V).view(B, N_V, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(E_I).view(B, L_I, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(E_I).view(B, L_I, self.n_heads, self.d_k).transpose(1, 2)

        # --- Scaled dot-product logits: (F_V W_Q)(E_I W_K)^T / sqrt(d_k) ---
        # Scale is 1/sqrt(d_k) with d_k the PER-HEAD dim (= 64), per paper.
        logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # logits: (B, n_heads, N_V, L_I)

        if key_padding_mask is not None:
            # True = padded instruction token -> exclude from softmax.
            logits = logits.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        # --- softmax over instruction positions, then weight E_I W_V. ---
        attn = torch.softmax(logits, dim=-1)          # (B, H, N_V, L_I)
        context = torch.matmul(attn, V)               # (B, H, N_V, d_k)

        # --- Merge heads and project back to d (output projection). ----
        context = (
            context.transpose(1, 2).contiguous().view(B, N_V, self.inner_dim)
        )
        ica_out = self.W_O(context)                   # (B, N_V, d)

        # --- Gated residual: F'_V = F_V + tanh(g) * ICA(F_V, E_I). ------
        gate = torch.tanh(self.g)                     # 0 exactly at init
        return F_V + gate * ica_out


if __name__ == "__main__":
    torch.manual_seed(0)

    for d, backbone in ((4096, "7B"), (5120, "13B")):
        B, N_V, L_I = 2, 576, 32  # 576 = LLaVA-style visual tokens; L_I ~= 32
        ica = InstructionConditionedAttention(d=d, n_heads=8, d_k=64)

        F_V = torch.randn(B, N_V, d)
        E_I = torch.randn(B, L_I, d)

        # --- 1) Identity at initialization (gate = tanh(0) = 0). --------
        with torch.no_grad():
            F_prime_V = ica(F_V, E_I)
        assert F_prime_V.shape == F_V.shape, (
            f"shape mismatch: {tuple(F_prime_V.shape)} vs {tuple(F_V.shape)}"
        )
        assert torch.allclose(F_prime_V, F_V, atol=1e-5), (
            "F'_V must equal F_V within 1e-5 at initialization (gate=0)"
        )
        max_dev = (F_prime_V - F_V).abs().max().item()

        # --- 2) key_padding_mask path also preserves identity at init. --
        pad_mask = torch.zeros(B, L_I, dtype=torch.bool)
        pad_mask[:, L_I // 2 :] = True  # pad the second half of E_I
        with torch.no_grad():
            F_prime_masked = ica(F_V, E_I, key_padding_mask=pad_mask)
        assert torch.allclose(F_prime_masked, F_V, atol=1e-5)

        # --- 3) With a non-zero gate the block must actually mix E_I. ---
        with torch.no_grad():
            ica.g.fill_(1.0)  # gate = tanh(1) ~= 0.7616
            F_prime_open = ica(F_V, E_I)
            ica.g.zero_()     # restore init state
        assert not torch.allclose(F_prime_open, F_V, atol=1e-3), (
            "with gate != 0 the output must differ from F_V"
        )

        n_params = sum(p.numel() for p in ica.parameters())
        print(
            f"[{backbone}] d={d}  F_V {tuple(F_V.shape)}  E_I {tuple(E_I.shape)}"
            f"  ->  F'_V {tuple(F_prime_V.shape)}"
        )
        print(
            f"[{backbone}] identity-at-init max |F'_V - F_V| = {max_dev:.3e}"
            f"  (gate = tanh(0) = 0);  params = {n_params:,}"
        )

    # --- 4) L_I cap = 64 is enforced. -----------------------------------
    ica_small = InstructionConditionedAttention(d=128)
    try:
        ica_small(torch.randn(1, 4, 128), torch.randn(1, 65, 128))
        raise AssertionError("L_I=65 should have been rejected (cap 64)")
    except ValueError:
        print("[cap] L_I > 64 correctly rejected (paper cap enforced)")

    print("ICA smoke test passed: identity at init, masking, gating, shapes OK.")
