
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REAL_ENV_HINT = (
    "The real-VLM path needs transformers/bitsandbytes/peft. Run it with:\n"
    "  HF_HOME=/data3/.cache/huggingface HF_HUB_OFFLINE=1 \\\n"
    "      /data/syupoh/anaconda3/bin/python3 <script> ...\n")

from transformers import (AutoProcessor, BitsAndBytesConfig,  # noqa: E402
                          LlavaForConditionalGeneration)

from ais import AdaptiveInstructionSelector, extract_first_answer_logits  # noqa: E402
from ica import InstructionConditionedAttention                           # noqa: E402

DEFAULT_MODEL = "llava-hf/llava-1.5-7b-hf"
IGNORE_INDEX = -100
STAGE_HEADERS = {
    "grounding": "Grounding:",
    "reasoning": "Reasoning:",
    "synthesis": "Synthesis:",
}
ANSWER_HEADER = "Answer:"


def _peak_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2 ** 30
    return 0.0


class RealVLM(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cuda",
                 lora: bool = False, grad_checkpoint: bool = True,
                 max_target_tokens: int = 96) -> None:
        super().__init__()
        self.device_ = device
        self.max_target_tokens = max_target_tokens

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, quantization_config=bnb, dtype=torch.bfloat16,
            device_map={"": 0} if device == "cuda" else None,
        )
        self.model.config.use_cache = False  # training default
        for p in self.model.parameters():
            p.requires_grad_(False)          # base VLM is FROZEN
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "right"
        self.pad_id = self.tokenizer.pad_token_id
        self.image_token_id = self.model.config.image_token_id

        # Trainable head: the imported ICA block at the LM hidden size.
        d = self.model.config.text_config.hidden_size  # 4096 for the 7B
        self.d = d
        self.ica = InstructionConditionedAttention(d=d, n_heads=8, d_k=64)
        self.ica.float().to(device)

        self.lora = lora
        if lora:
            from peft import LoraConfig, inject_adapter_in_model
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=r".*language_model.*\.(q_proj|v_proj)$")
            inject_adapter_in_model(cfg, self.model)
            n_lora = sum(p.numel() for p in self.model.parameters()
                         if p.requires_grad)
            print(f"[real] LoRA enabled on LM q_proj/v_proj (r=16): "
                  f"{n_lora:,} trainable LM params")

        self.grad_checkpoint = grad_checkpoint
        if grad_checkpoint:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            # HF layers apply checkpointing only in train() mode; llama-2
            # has no dropout so train() mode is numerically identical.
            self.model.model.language_model.train()

        n_ica = sum(p.numel() for p in self.ica.parameters())
        print(f"[real] {model_name} 4-bit NF4 (bf16 compute) on {device}; "
              f"base FROZEN; trainable ICA d={d} n_heads=8 d_k=64 "
              f"params={n_ica:,} (float32)"
              + (" + LoRA" if lora else ""))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_image(self, path: Optional[str], pil) -> "Image.Image":
        from PIL import Image
        if pil is not None:
            return pil.convert("RGB")
        if path and os.path.isfile(path):
            return Image.open(path).convert("RGB")
        print(f"[real][warn] image missing ({path!r}); using gray dummy",
              file=sys.stderr)
        return Image.new("RGB", (336, 336), (127, 127, 127))

    def _images_of(self, batch: Dict) -> List:
        n = len(batch["id"])
        pils = batch.get("image") or [None] * n
        return [self._load_image(p, im)
                for p, im in zip(batch["image_path"], pils)]

    def _encode_prompts(self, texts: Sequence[str], images: Sequence
                        ) -> Tuple[List[List[int]], torch.Tensor]:
        ids_list: List[List[int]] = []
        pixels: List[torch.Tensor] = []
        for t, im in zip(texts, images):
            out = self.processor(text=t, images=im, return_tensors="pt")
            ids_list.append(out["input_ids"][0].tolist())
            pixels.append(out["pixel_values"])
        return ids_list, torch.cat(pixels, dim=0).to(self.device_,
                                                     torch.bfloat16)

    def _pad_batch(self, ids_list: List[List[int]],
                   labels_list: Optional[List[List[int]]] = None):
        L = max(len(r) for r in ids_list)
        B = len(ids_list)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, r in enumerate(ids_list):
            ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, :len(r)] = 1
        ids = ids.to(self.device_)
        attn = attn.to(self.device_)
        if labels_list is None:
            return ids, attn
        labels = torch.full((B, L), IGNORE_INDEX, dtype=torch.long)
        for i, r in enumerate(labels_list):
            labels[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        return ids, attn, labels.to(self.device_)

    def _splice_embeds(self, ids: torch.Tensor,
                       f_prime: torch.Tensor) -> torch.Tensor:
        embeds = self.model.get_input_embeddings()(ids)          # (B,T,d) bf16
        mask = (ids == self.image_token_id)
        n_img = int(mask.sum())
        if n_img != f_prime.shape[0] * f_prime.shape[1]:
            raise RuntimeError(
                f"image-token count {n_img} != refined features "
                f"{tuple(f_prime.shape[:2])} — placeholder expansion mismatch")
        src = f_prime.to(embeds.dtype).reshape(-1, embeds.shape[-1])
        return embeds.masked_scatter(
            mask.unsqueeze(-1).expand_as(embeds), src)

    def visual_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.get_image_features(pixel_values=pixel_values,
                                                return_dict=True)
            feats = out.pooler_output  # transformers 5.3: list of (576, d)
            if isinstance(feats, (list, tuple)):
                feats = torch.stack(list(feats), dim=0)
        return feats.detach()

    def refine_visuals(self, pixel_values: torch.Tensor,
                       instructions: Sequence[str]) -> torch.Tensor:
        F_V = self.visual_features(pixel_values).float()         # (B,576,d)
        enc = self.tokenizer(list(instructions), add_special_tokens=False,
                             truncation=True,
                             max_length=self.ica.MAX_INSTRUCTION_LEN,
                             padding=True, return_tensors="pt")
        instr_ids = enc["input_ids"].to(self.device_)
        pad_mask = enc["attention_mask"].to(self.device_).eq(0)  # True=pad
        with torch.no_grad():
            E_I = self.model.get_input_embeddings()(instr_ids).float()
        return self.ica(F_V, E_I.detach(), key_padding_mask=pad_mask)

    # ------------------------------------------------------------------
    # stub-compatible interface used by the train loop
    # ------------------------------------------------------------------
    def base_context(self, batch: Dict) -> Dict:
        images = self._images_of(batch)
        texts = [f"USER: <image>\n{q} ASSISTANT:" for q in batch["question"]]
        ids_list, pixel_values = self._encode_prompts(texts, images)
        ids, attn = self._pad_batch(ids_list)
        idx = torch.tensor([len(r) - 1 for r in ids_list], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn,
                "pixel_values": pixel_values, "answer_start": idx}

    def first_answer_logits(self, ctx: Dict) -> torch.Tensor:
        out = self.model(input_ids=ctx["input_ids"],
                         attention_mask=ctx["attention_mask"],
                         pixel_values=ctx["pixel_values"],
                         use_cache=False)
        logits = out.logits.float().cpu()                        # (B,T,V)
        return extract_first_answer_logits(logits, ctx["answer_start"])

    def conditioned_forward(self, batch: Dict,
                            instructions: Sequence[str]) -> Tuple[Dict, Dict]:
        images = self._images_of(batch)
        main_texts = [f"USER: <image>\n{ins}\n{q} ASSISTANT:"
                      for ins, q in zip(instructions, batch["question"])]
        ica_texts = [f"USER: <image>\n{q} ASSISTANT:"
                     for q in batch["question"]]
        main_ids, pixel_values = self._encode_prompts(main_texts, images)
        ica_ids, _ = self._encode_prompts(ica_texts, images)
        f_prime = self.refine_visuals(pixel_values, instructions)
        ctx_main = {"prompt_ids": main_ids, "f_prime": f_prime}
        ctx_ica = {"prompt_ids": ica_ids, "f_prime": f_prime}
        return ctx_main, ctx_ica

    def text_ce(self, ctx: Dict, texts: Sequence[str],
                tag: str = "answer") -> torch.Tensor:
        header = None if tag == "answer" else STAGE_HEADERS[tag]
        ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        for prompt_ids, text in zip(ctx["prompt_ids"], texts):
            prefix = list(prompt_ids)
            if header is not None:
                prefix += self.tokenizer(" " + header,
                                         add_special_tokens=False)["input_ids"]
            target = self.tokenizer(" " + str(text),
                                    add_special_tokens=False)["input_ids"]
            target = target[: self.max_target_tokens]
            target = target + [self.tokenizer.eos_token_id]
            ids_list.append(prefix + target)
            labels_list.append([IGNORE_INDEX] * len(prefix) + target)
        ids, attn, labels = self._pad_batch(ids_list, labels_list)
        embeds = self._splice_embeds(ids, ctx["f_prime"])
        out = self.model(inputs_embeds=embeds, attention_mask=attn,
                         labels=labels, use_cache=False)
        return out.loss

    # ------------------------------------------------------------------
    # generation (real inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_text(self, embeds: torch.Tensor, attn: torch.Tensor,
                      max_new_tokens: int) -> Tuple[str, List[int]]:
        out = self.model.generate(
            inputs_embeds=embeds, attention_mask=attn,
            max_new_tokens=max_new_tokens, do_sample=False,
            use_cache=True, pad_token_id=self.pad_id)
        new_ids = out[0].tolist()  # inputs_embeds => only new tokens returned
        stop = {self.pad_id, self.tokenizer.eos_token_id}
        new_ids = [i for i in new_ids if i not in stop]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return text, new_ids

    def append_tokens(self, embeds: torch.Tensor, attn: torch.Tensor,
                      ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not ids:
            return embeds, attn
        t = torch.tensor([ids], dtype=torch.long, device=self.device_)
        with torch.no_grad():
            e = self.model.get_input_embeddings()(t)
        return (torch.cat([embeds, e.to(embeds.dtype)], dim=1),
                torch.cat([attn, torch.ones_like(t)], dim=1))


def build_real_model(args) -> RealVLM:
    if not torch.cuda.is_available():
        raise RuntimeError("Real-VLM path requires a CUDA GPU.")
    torch.cuda.reset_peak_memory_stats()
    return RealVLM(model_name=args.model, lora=getattr(args, "lora", False),
                   grad_checkpoint=not getattr(args, "no_grad_checkpoint",
                                               False))


# ---------------------------------------------------------------------------
# Real inference (called from infer_instructcot.py's non-stub branch)
# ---------------------------------------------------------------------------

COT_STAGES = ("grounding", "reasoning", "synthesis")


def run_infer(args) -> int:
    from dataset import InstructCoTDataset

    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()

    ds = InstructCoTDataset(source=args.source, data_path=args.data,
                            split=args.split, limit=args.limit)
    n = min(len(ds), args.max_samples)
    samples = [ds[i] for i in range(n)]
    print(f"[data] source={args.source} rows={len(ds)} using n={n}")

    model = RealVLM(model_name=args.model, lora=False, grad_checkpoint=False)
    model.eval()

    selector = AdaptiveInstructionSelector()
    ckpt_path = getattr(args, "ica_ckpt", None)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.ica.load_state_dict(ckpt["ica_state_dict"])
        model.ica.float().to(model.device_)
        if ckpt.get("tau1") is not None and ckpt.get("tau2") is not None:
            selector.tau1 = float(ckpt["tau1"])
            selector.tau2 = float(ckpt["tau2"])
            print(f"[ica] loaded checkpoint {ckpt_path} "
                  f"(reusing saved tau1={selector.tau1:.4f} "
                  f"tau2={selector.tau2:.4f} bits — no recalibration)")
        else:
            print(f"[ica] loaded checkpoint {ckpt_path} (no taus stored)")

    # --- 1) AIS scoring: instruction-free frozen-base entropy -------------
    scores: List[float] = []
    with torch.no_grad():
        for s in samples:
            batch1 = {k: [v] for k, v in s.items()}
            logits = model.first_answer_logits(model.base_context(batch1))
            scores.append(float(selector.compute_entropy(logits)[0]))
    if selector.tau1 is None:
        tau1, tau2 = selector.calibrate(scores, p_low=28, p_high=72)
        print(f"[ais] calibrated once on n={n} scores: tau1(P28)={tau1:.4f} "
              f"tau2(P72)={tau2:.4f} bits")
    tau1, tau2 = selector.tau1, selector.tau2

    # --- 2..5) per-sample routing + refined embeds + gated generation -----
    from train_instructcot import INSTRUCTION_TEMPLATES
    levels: List[int] = []
    n_skipped = 0
    for i, s in enumerate(samples):
        score = scores[i]
        level = selector.assign_level(score)
        levels.append(level)
        instruction = (s.get("instruction_text")
                       or selector.select_instruction(level,
                                                      INSTRUCTION_TEMPLATES))
        batch1 = {k: [v] for k, v in s.items()}
        with torch.no_grad():
            ctx_main, _ = model.conditioned_forward(batch1, [instruction])
            ids, attn = model._pad_batch(ctx_main["prompt_ids"])
            embeds = model._splice_embeds(ids, ctx_main["f_prime"])

            skip_cot = bool(score < tau1) or getattr(args, "no_cot", False)
            gate = "SKIP-CoT" if skip_cot else "3-stage CoT"
            cot_out: Dict[str, str] = {}
            if skip_cot:
                n_skipped += 1
                answer, _ = model.generate_text(
                    embeds, attn, max_new_tokens=args.answer_new_tokens)
            else:
                for stage in COT_STAGES:
                    hdr = model.tokenizer(" " + STAGE_HEADERS[stage],
                                          add_special_tokens=False)["input_ids"]
                    embeds, attn = model.append_tokens(embeds, attn, hdr)
                    text, new_ids = model.generate_text(
                        embeds, attn, max_new_tokens=args.stage_new_tokens)
                    cot_out[stage] = text
                    embeds, attn = model.append_tokens(embeds, attn, new_ids)
                hdr = model.tokenizer(" " + ANSWER_HEADER,
                                      add_special_tokens=False)["input_ids"]
                embeds, attn = model.append_tokens(embeds, attn, hdr)
                answer, _ = model.generate_text(
                    embeds, attn, max_new_tokens=args.answer_new_tokens)

        gold = s.get("final_answer") or s.get("gold_answer")
        print(f"[sample {i + 1}/{n}] id={s['id']} entropy={score:.3f} bits "
              f"-> L{level} | {gate}")
        print(f"  instruction: {instruction[:70]}...")
        for stage in COT_STAGES:
            if stage in cot_out:
                print(f"  cot.{stage}: {cot_out[stage]!r}")
        print(f"  generated: {answer!r}  (gold: {str(gold)!r})")
        assert isinstance(answer, str) and len(answer) > 0, \
            "empty decoded answer"

    lvl_counts = {lv: levels.count(lv) for lv in (1, 2, 3)}
    print(f"[summary] levels L1/L2/L3 = {lvl_counts[1]}/{lvl_counts[2]}/"
          f"{lvl_counts[3]} | CoT skipped {n_skipped}/{n} "
          f"(gate: entropy < tau1={tau1:.4f} bits) | "
          f"peak_vram={_peak_vram_gb():.2f}GiB")
    if getattr(args, "smoke", False):
        print("[smoke] REAL INFER SMOKE PASSED: AIS routing, ICA-refined "
              "embeds, entropy CoT-skip gate, and real generation ran "
              "end-to-end on llava-1.5-7b-hf 4-bit (plumbing verification "
              "only — decoded text is NOT an evaluated result).")
    return 0
