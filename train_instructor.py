
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Same-directory imports must work no matter the caller's cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ais import AdaptiveInstructionSelector, extract_first_answer_logits  # noqa: E402
from ica import InstructionConditionedAttention                           # noqa: E402
from pcc import ProgressiveCoTCurriculum, instructcot_loss                # noqa: E402
from dataset import InstructCoTDataset, collate_fn                        # noqa: E402

PAD_ID = 0  # char-level pad id (ignored in CE)

# AIS level -> fallback instruction templates (paper Sec 3.3 semantics),
# used when a sample carries no teacher-written instruction_text.
INSTRUCTION_TEMPLATES: Dict[int, str] = {
    1: "This is a visual question answering task. Answer with a short phrase.",
    2: ("This is a visual question answering task. Bring relevant domain "
        "knowledge to bear, then answer with a short phrase."),
    3: ("This is a visual reasoning task. Identify the required reasoning "
        "capabilities, apply them in order, then answer with a short phrase."),
}


class CharTokenizer:

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len: int) -> List[int]:
        ids = [max(1, ord(c) % self.vocab_size) for c in text[:max_len]]
        return ids or [1]

    def batch_encode(self, texts: Sequence[str], max_len: int) -> torch.Tensor:
        rows = [self.encode(t, max_len) for t in texts]
        L = max(len(r) for r in rows)
        out = torch.full((len(rows), L), PAD_ID, dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return out


class StubVLM(nn.Module):

    def __init__(self, d: int = 64, vocab_size: int = 128,
                 n_vision_tokens: int = 8):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.n_vision_tokens = n_vision_tokens
        self.tokenizer = CharTokenizer(vocab_size)

        self.token_embed = nn.Embedding(vocab_size, d, padding_idx=PAD_ID)
        self.vision_base = nn.Parameter(torch.randn(n_vision_tokens, d) * 0.02)
        self.image_hash_embed = nn.Embedding(64, d)
        self.vision_mlp = nn.Sequential(  # "a couple of layers"
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

        # ICA: paper defaults n_heads=8, d_k=64 (inner dim 512) on width d.
        self.ica = InstructionConditionedAttention(d=d, n_heads=8, d_k=64)

        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.gru = nn.GRU(d, d, batch_first=True)
        self.lm_head = nn.Linear(d, vocab_size)

    # ---- encoders ---------------------------------------------------------
    def encode_images(self, image_paths: Sequence[Optional[str]]) -> torch.Tensor:
        ids = torch.tensor(
            [abs(hash(p or "null_image")) % 64 for p in image_paths],
            dtype=torch.long)
        img_vec = self.image_hash_embed(ids)                       # (B, d)
        base = self.vision_base.unsqueeze(0).expand(len(ids), -1, -1)
        return self.vision_mlp(base + img_vec[:, None, :])         # (B,N_V,d)

    def embed_text(self, texts: Sequence[str], max_len: int = 64):
        ids = self.tokenizer.batch_encode(texts, max_len)          # (B, L)
        return self.token_embed(ids), ids.eq(PAD_ID), ids          # emb, pad mask

    @staticmethod
    def _masked_mean(emb: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        keep = (~pad).float().unsqueeze(-1)
        return (emb * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    # ---- LM heads ---------------------------------------------------------
    def sequence_logits(self, context: torch.Tensor,
                        target_ids: torch.Tensor) -> torch.Tensor:
        B, T = target_ids.shape
        tgt_emb = self.token_embed(target_ids)
        inp = torch.cat([self.bos.expand(B, 1, -1), tgt_emb[:, :-1]], dim=1)
        out, _ = self.gru(inp, context.unsqueeze(0).contiguous())
        return self.lm_head(out)                                   # (B,T,V)

    def text_ce(self, context: torch.Tensor, texts: Sequence[str],
                max_len: int = 64, tag: str = "answer") -> torch.Tensor:
        ids = self.tokenizer.batch_encode(texts, max_len)
        logits = self.sequence_logits(context, ids)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               ids.reshape(-1), ignore_index=PAD_ID)

    def first_answer_logits(self, context: torch.Tensor) -> torch.Tensor:
        B = context.shape[0]
        out, _ = self.gru(self.bos.expand(B, 1, -1),
                          context.unsqueeze(0).contiguous())
        logits_seq = self.lm_head(out)                             # (B,1,V)
        idx = torch.zeros(B, dtype=torch.long)
        return extract_first_answer_logits(logits_seq, idx)        # (B,V)

    # ---- context builders -------------------------------------------------
    def base_context(self, batch: Dict) -> torch.Tensor:
        F_V = self.encode_images(batch["image_path"])
        q_emb, q_pad, _ = self.embed_text(batch["question"])
        return 0.5 * (F_V.mean(1) + self._masked_mean(q_emb, q_pad))

    def conditioned_forward(self, batch: Dict, instructions: Sequence[str]):
        F_V = self.encode_images(batch["image_path"])              # (B,N_V,d)
        E_I, i_pad, _ = self.embed_text(list(instructions),
                                        max_len=self.ica.MAX_INSTRUCTION_LEN)
        F_prime_V = self.ica(F_V, E_I, key_padding_mask=i_pad)     # ICA block
        q_emb, q_pad, _ = self.embed_text(batch["question"])
        i_mean = self._masked_mean(E_I, i_pad)
        q_mean = self._masked_mean(q_emb, q_pad)
        context = (F_prime_V.mean(1) + i_mean + q_mean) / 3.0
        # L_ICA conditions on (F'_V, q, I) with the refined visual features
        # dominant, so the auxiliary CE pushes gradient through the ICA gate.
        ica_context = F_prime_V.mean(1) + 0.5 * (i_mean + q_mean)
        return context, ica_context


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def route_instructions(model: StubVLM,
                       selector: AdaptiveInstructionSelector,
                       batch: Dict) -> (List[int], List[str]):
    with torch.no_grad():  # frozen-base scoring: no gradients
        logits = model.first_answer_logits(model.base_context(batch))
        scores = selector.compute_entropy(logits)                  # bits
    if selector.tau1 is None:
        tau1, tau2 = selector.calibrate(scores, p_low=28, p_high=72)
        print(f"[ais] calibrated once on {len(scores)} scores: "
              f"tau1(P28)={tau1:.4f} bits tau2(P72)={tau2:.4f} bits")
    levels = [selector.assign_level(float(s)) for s in scores]
    instructions = []
    for lv, given in zip(levels, batch.get("instruction_text",
                                           [None] * len(levels))):
        # Teacher-annotated instruction_text (InstructCoT-200K rows) wins;
        # otherwise fall back to the routed level's template.
        instructions.append(
            given if given else
            selector.select_instruction(lv, INSTRUCTION_TEMPLATES))
    return levels, instructions, scores


def _build_real_model(args):
    try:
        import real_vlm
    except ImportError as e:
        print(f"[error] real-VLM import failed: {e}\n"
              "The real path needs transformers/bitsandbytes/peft. Run:\n"
              "  HF_HOME=/data3/.cache/huggingface HF_HUB_OFFLINE=1 \\\n"
              "      /data/syupoh/anaconda3/bin/python3 "
              "train_instructcot.py ...\n"
              "(or pass --stub for the CPU plumbing check)", file=sys.stderr)
        raise SystemExit(2)
    return real_vlm.build_real_model(args)


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    if args.batch_size is None:  # stub keeps its old default of 5
        args.batch_size = 5 if args.stub else 2

    ds = InstructCoTDataset(source=args.source, data_path=args.data,
                            limit=args.limit)
    loader = DataLoader(ds, batch_size=min(args.batch_size, len(ds)),
                        shuffle=True, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(42))
    print(f"[data] source={args.source} rows={len(ds)} "
          f"batch_size={min(args.batch_size, len(ds))}")

    if args.stub:
        model = StubVLM(d=args.d)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[model] StubVLM d={args.d} params={n_params:,} (CPU)")
    else:
        model = _build_real_model(args)
        # snapshot trainable ICA tensors to verify they actually change
        ica_before = {k: v.detach().clone().cpu()
                      for k, v in model.ica.state_dict().items()}

    selector = AdaptiveInstructionSelector()
    pcc = ProgressiveCoTCurriculum()
    trainable = [p for p in model.parameters() if p.requires_grad]
    lr = args.lr if args.lr is not None else (1e-3 if args.stub else 1e-4)
    optim = torch.optim.AdamW(trainable, lr=lr)

    steps = min(args.steps, 3) if args.smoke and args.steps > 3 else args.steps
    it = iter(loader)
    first_loss = last_loss = None
    for step in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        # --- 1) AIS: route instruction level from frozen-base entropy -----
        levels, instructions, scores = route_instructions(model, selector, batch)

        # --- 3) PCC: curriculum mode selection ----------------------------
        # progress = fraction of training completed at step start; a 3-step
        # smoke visits p=0, 1/3, 2/3 -> phase 1 (direct), phase 2
        # (rationale), phase 3 (full CoT), exercising every branch.
        progress = step / max(steps, 1)
        w = pcc.schedule(progress)
        mode = rng.choice(pcc.MODES, p=[w["direct"], w["rationale"],
                                        w["full_cot"]])
        answers = [str(a) for a in
                   (batch.get("final_answer") or batch["gold_answer"])]
        cot_steps = batch.get("cot_steps")
        if cot_steps is None and mode != "direct":
            mode = "direct"  # sources without CoT annotation train direct

        # --- 2+3) ICA conditioning + composite loss + backward ------------
        def _step_losses(bt, instrs, ans, steps_rows):
            context, ica_context = model.conditioned_forward(bt, instrs)
            a_ce = model.text_ce(context, ans, tag="answer")
            i_ce = model.text_ce(ica_context, ans, tag="answer")  # L_ICA aux
            if mode == "full_cot":
                tags = ("grounding", "reasoning", "synthesis")
                s_ces = [model.text_ce(context,
                                       [s[k] for s in _rows(steps_rows)],
                                       tag=tags[k])
                         for k in range(3)]
            elif mode == "rationale":
                s_ces = [model.text_ce(context,
                                       [s[2] for s in _rows(steps_rows)],
                                       tag="synthesis")]
            else:
                s_ces = []
            l = instructcot_loss(s_ces, a_ce, i_ce,
                                 alpha=0.3, beta=1.0, lam=0.1)
            # --- 4) backward + optimizer step -----------------------------
            optim.zero_grad(set_to_none=True)
            l.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optim.step()
            return a_ce, i_ce, s_ces, l, gn

        try:
            answer_ce, ica_ce, step_ces, loss, grad_norm = _step_losses(
                batch, instructions, answers, cot_steps)
        except torch.cuda.OutOfMemoryError:
            # defensive OOM fallback (shared GPU): halve the batch, retry once
            half = max(1, len(answers) // 2)
            print(f"[oom] CUDA OOM at batch={len(answers)}; retrying with "
                  f"batch={half}", file=sys.stderr)
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            batch = {k: (v[:half] if isinstance(v, list) else v)
                     for k, v in batch.items()}
            instructions = instructions[:half]
            answers = answers[:half]
            cot_steps = cot_steps[:half] if cot_steps is not None else None
            answer_ce, ica_ce, step_ces, loss, grad_norm = _step_losses(
                batch, instructions, answers, cot_steps)

        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        lvl_str = "".join(str(l) for l in levels)
        vram = ""
        if not args.stub and torch.cuda.is_available():
            vram = (f" peak_vram="
                    f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GiB")
        print(f"[step {step + 1}/{steps}] p={progress:.2f} "
              f"phase={pcc.phase(progress)} mode={mode:9s} "
              f"w=(d={w['direct']:.2f},r={w['rationale']:.2f},"
              f"f={w['full_cot']:.2f}) ais_levels={lvl_str} "
              f"entropy[bits]=({scores.min():.2f}..{scores.max():.2f}) | "
              f"answer_ce={answer_ce.item():.4f} ica_ce={ica_ce.item():.4f} "
              f"step_ces={[round(s.item(), 4) for s in step_ces]} "
              f"loss={loss.item():.4f} grad_norm={grad_norm:.4f}{vram}")
        assert torch.isfinite(loss), "non-finite loss"
        assert grad_norm > 0, "zero grad_norm: backward path broken"

    print(f"[done] {steps} optimizer steps | loss {first_loss:.4f} -> "
          f"{last_loss:.4f} | grad flow verified (grad_norm > 0 every step)")

    if not args.stub:
        # verify the trainable ICA weights actually changed (state-dict diff)
        diffs = {k: float((v.detach().cpu() - ica_before[k]).abs().max())
                 for k, v in model.ica.state_dict().items()}
        changed = [k for k, dv in diffs.items() if dv > 0]
        print("[ica] weight change (max |after - before| per tensor): "
              + ", ".join(f"{k}={dv:.3e}" for k, dv in diffs.items()))
        assert changed, "ICA weights did NOT change during training"
        print(f"[ica] {len(changed)}/{len(diffs)} ICA tensors changed "
              "(trainable head verified)")

    if args.save_ica:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_ica)),
                    exist_ok=True)
        torch.save({
            "ica_state_dict": {k: v.detach().cpu()
                               for k, v in model.ica.state_dict().items()},
            "tau1": selector.tau1,
            "tau2": selector.tau2,
            "config": {"d": model.ica.d, "n_heads": model.ica.n_heads,
                       "d_k": model.ica.d_k,
                       "model": getattr(args, "model", None),
                       "stub": bool(args.stub)},
        }, args.save_ica)
        print(f"[ckpt] saved ICA state + taus (tau1={selector.tau1:.4f}, "
              f"tau2={selector.tau2:.4f}) -> {args.save_ica}")

    if args.smoke:
        if args.stub:
            print("[smoke] TRAIN SMOKE PASSED: AIS->ICA->PCC->loss->backward "
                  "ran end-to-end on CPU with synthetic data (stub losses are "
                  "plumbing checks, not results).")
        else:
            print("[smoke] REAL TRAIN SMOKE PASSED: AIS->ICA->PCC->loss->"
                  "backward ran end-to-end on llava-1.5-7b-hf 4-bit with the "
                  "trainable ICA head (smoke losses are plumbing checks, not "
                  "trained-model results).")
    return 0


def _rows(cot_steps_batch: List[List[str]]) -> List[List[str]]:
    return cot_steps_batch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=str, default=None,
                   help="local JSONL file OR directory of *.jsonl "
                        "(instructcot_200k) / HF cache dir override")
    p.add_argument("--source", type=str, default="instructcot_200k",
                   help="registered format adapter key")
    p.add_argument("--stub", action="store_true",
                   help="use the tiny CPU StubVLM (no GPU/network/datasets)")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--d", type=int, default=64, help="stub model width")
    p.add_argument("--smoke", action="store_true",
                   help="smoke mode: cap steps at 3, assert grad flow")
    p.add_argument("--batch-size", type=int, default=None,
                   help="default: 5 for --stub, 2 for the real VLM")
    p.add_argument("--limit", type=int, default=None,
                   help="cap dataset rows (smoke tests)")
    # --- real-VLM path options (ignored by --stub) ---
    p.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                   help="HF model id for the real path (local cache, "
                        "HF_HUB_OFFLINE=1)")
    p.add_argument("--lora", action="store_true",
                   help="also train LoRA r=16 on LM q_proj/v_proj "
                        "(default OFF; ICA-only otherwise)")
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 stub / 1e-4 real")
    p.add_argument("--save-ica", type=str, default=None,
                   help="save {ica state_dict, tau1/tau2, config} here")
    p.add_argument("--no-grad-checkpoint", action="store_true",
                   help="disable LM gradient checkpointing (real path)")
    return p


if __name__ == "__main__":
    sys.exit(train(build_parser().parse_args()))
