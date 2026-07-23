
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ais import AdaptiveInstructionSelector                    # noqa: E402
from dataset import InstructCoTDataset, collate_fn             # noqa: E402
from train_instructcot import (                                # noqa: E402
    INSTRUCTION_TEMPLATES, PAD_ID, StubVLM)

COT_STAGES = ("grounding", "reasoning", "synthesis")


@torch.no_grad()
def greedy_decode(model: StubVLM, context: torch.Tensor,
                  max_new_tokens: int = 12) -> List[str]:
    B = context.shape[0]
    h = context.unsqueeze(0).contiguous()
    inp = model.bos.expand(B, 1, -1)
    out_ids = torch.zeros(B, max_new_tokens, dtype=torch.long)
    for t in range(max_new_tokens):
        out, h = model.gru(inp, h)
        nxt = model.lm_head(out[:, -1]).argmax(dim=-1)          # (B,)
        out_ids[:, t] = nxt
        inp = model.token_embed(nxt).unsqueeze(1)
    texts = []
    for row in out_ids:
        chars = [chr(int(i)) for i in row
                 if int(i) != PAD_ID and 32 <= int(i) < 127]
        texts.append("".join(chars))
    return texts


def _infer_real(args: argparse.Namespace) -> int:``
    try:
        import real_vlm
    except ImportError as e:
        print(f"[error] real-VLM import failed: {e}\n"
              "The real path needs transformers/bitsandbytes/peft. Run:\n"
              "  HF_HOME=/data3/.cache/huggingface HF_HUB_OFFLINE=1 \\\n"
              "      /data/syupoh/anaconda3/bin/python3 "
              "infer_instructcot.py ...\n"
              "(or pass --stub for the CPU plumbing check)", file=sys.stderr)
        return 2
    return real_vlm.run_infer(args)


@torch.no_grad()
def infer(args: argparse.Namespace) -> int:
    if not args.stub:
        return _infer_real(args)

    torch.manual_seed(42)

    ds = InstructCoTDataset(source=args.source, data_path=args.data,
                            split=args.split, limit=args.limit)
    n = min(len(ds), args.max_samples)
    batch: Dict = collate_fn([ds[i] for i in range(n)])
    print(f"[data] source={args.source} rows={len(ds)} using n={n}")

    # Stub-only variance fix: HF benchmark rows have image_path=None (their PIL
    # image is never decoded by the stub), so every such row would hash to the
    # same "null_image" vision embedding -- substitute a stable per-sample key.
    batch["image_path"] = [p if p else f"img::{batch['id'][i]}"
                           for i, p in enumerate(batch["image_path"])]

    model = StubVLM(d=args.d)
    model.eval()
    print(f"[model] StubVLM d={args.d} (CPU, eval mode)")

    # --- 1) AIS scoring on the instruction-free base forward --------------
    selector = AdaptiveInstructionSelector()
    base_ctx = model.base_context(batch)
    logits = model.first_answer_logits(base_ctx)   # uses extract_first_answer_logits
    scores = selector.compute_entropy(logits)      # bits
    tau1, tau2 = selector.calibrate(scores, p_low=28, p_high=72)
    print(f"[ais] calibrated once on n={n} scores: tau1(P28)={tau1:.4f} "
          f"tau2(P72)={tau2:.4f} bits")

    # --- 2) route levels + instructions -----------------------------------
    levels = [selector.assign_level(float(s)) for s in scores]
    annotated = batch.get("instruction_text", [None] * n)
    instructions = [
        given if given else
        selector.select_instruction(lv, INSTRUCTION_TEMPLATES)
        for lv, given in zip(levels, annotated)]

    # --- 3) ICA-conditioned forward ---------------------------------------
    context, _ = model.conditioned_forward(batch, instructions)

    # --- 4+5) entropy CoT-skip gate + greedy decode -----------------------
    # Gate: skip CoT when s(v,q) < tau1 (routed Level 1 == low complexity).
    skip_cot = [bool(float(s) < tau1) or args.no_cot for s in scores]
    answers = greedy_decode(model, context, max_new_tokens=args.max_new_tokens)
    cot_texts: List[Dict[str, str]] = [{} for _ in range(n)]
    need_cot = [i for i in range(n) if not skip_cot[i]]
    if need_cot:
        sub_ctx = context[need_cot]
        for stage in COT_STAGES:  # one decode pass per CoT stage
            stage_out = greedy_decode(model, sub_ctx,
                                      max_new_tokens=args.max_new_tokens)
            for j, i in enumerate(need_cot):
                cot_texts[i][stage] = stage_out[j]

    n_skipped = sum(skip_cot)
    for i in range(n):
        gold = (batch.get("final_answer") or batch["gold_answer"])[i]
        gate = "SKIP-CoT" if skip_cot[i] else "3-stage CoT"
        print(f"[sample {i + 1}/{n}] id={batch['id'][i]} "
              f"entropy={float(scores[i]):.3f} bits -> L{levels[i]} | {gate}")
        print(f"  instruction: {instructions[i][:70]}...")
        if not skip_cot[i]:
            for stage in COT_STAGES:
                print(f"  cot.{stage}: {cot_texts[i][stage]!r}")
        print(f"  generated: {answers[i]!r}  (gold: {str(gold)!r})")

    lvl_counts = {lv: levels.count(lv) for lv in (1, 2, 3)}
    print(f"[summary] levels L1/L2/L3 = {lvl_counts[1]}/{lvl_counts[2]}/"
          f"{lvl_counts[3]} | CoT skipped {n_skipped}/{n} "
          f"(gate: entropy < tau1={tau1:.4f} bits)")
    assert len(answers) == n and all(isinstance(a, str) for a in answers)
    if args.smoke:
        print("[smoke] INFER SMOKE PASSED: AIS routing, ICA conditioning, "
              "entropy CoT-skip gate, and greedy decode ran end-to-end on "
              "CPU (stub generations are plumbing checks, not results).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=str, default=None,
                   help="local JSONL file OR directory of *.jsonl "
                        "(instructcot_200k) / HF cache dir override")
    p.add_argument("--source", type=str, default="instructcot_200k")
    p.add_argument("--split", type=str, default=None,
                   help="dataset split (default: adapter's default split)")
    p.add_argument("--stub", action="store_true",
                   help="use the tiny CPU StubVLM (no GPU/network/datasets)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--d", type=int, default=64, help="stub model width")
    p.add_argument("--max-samples", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=12)
    p.add_argument("--no-cot", action="store_true",
                   help="force the CoT-skip gate for every sample")
    p.add_argument("--limit", type=int, default=None)
    # --- real-VLM path options (ignored by --stub) ---
    p.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                   help="HF model id for the real path (local cache, "
                        "HF_HUB_OFFLINE=1)")
    p.add_argument("--ica-ckpt", type=str, default=None,
                   help="ICA checkpoint from train --save-ica; also reuses "
                        "the saved tau1/tau2 (no recalibration)")
    p.add_argument("--stage-new-tokens", type=int, default=32,
                   help="real path: max new tokens per CoT stage")
    p.add_argument("--answer-new-tokens", type=int, default=16,
                   help="real path: max new tokens for the final answer")
    return p


if __name__ == "__main__":
    sys.exit(infer(build_parser().parse_args()))
