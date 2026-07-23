
from __future__ import annotations

import ast
import json
import os
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Small helpers (pure python, no heavy deps)
# ---------------------------------------------------------------------------

def _majority(answers: Sequence[Any]) -> str:
    flat: List[str] = []
    for a in answers:
        if isinstance(a, dict):
            a = a.get("answer", "")
        a = str(a).strip()
        if a:
            flat.append(a.lower())
    if not flat:
        return ""
    return Counter(flat).most_common(1)[0][0]


def _extract_tag(text: str, tag: str) -> str:
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    lo = text.find(open_t)
    if lo < 0:
        return ""
    lo += len(open_t)
    hi = text.find(close_t, lo)
    return text[lo:hi].strip() if hi >= 0 else text[lo:].strip()


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------
# An adapter is (load_fn, map_fn):
#   load_fn(data_path, split, limit) -> sequence of RAW records
#   map_fn(raw_record, index)        -> unified sample dict (pure, no I/O deps)
# HF `datasets` is imported ONLY inside the HF load_fns (lazy import).

FORMAT_ADAPTERS: Dict[str, Dict[str, Callable]] = {}


def register_adapter(name: str, load_fn: Callable, map_fn: Callable) -> None:
    FORMAT_ADAPTERS[name] = {"load": load_fn, "map": map_fn}


def _hf_loader(hf_path: str, config: Optional[str] = None,
               default_split: str = "train") -> Callable:
    def load(data_path: Optional[str], split: Optional[str], limit: Optional[int]):
        try:
            import datasets  # LAZY import -- only needed for HF sources
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                f"Loading '{hf_path}' requires the 'datasets' package "
                "(pip install datasets). The local 'instructcot_200k' JSONL "
                "path does not."
            ) from e
        sp = split or default_split
        args = [hf_path] + ([config] if config else [])
        ds = datasets.load_dataset(*args, split=sp,
                                   cache_dir=data_path or None)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        return ds
    return load


# --- vqav2: lmms-lab/VQAv2 (design doc Section 2.1; lmms-lab mirror is the
#     one present in the local HF cache -- validation split, n=214354) ------
def _map_vqav2(r: Dict, i: int) -> Dict:
    return {
        "id": f"vqav2_{r.get('question_id', i)}",
        "source": "vqav2",
        "image_path": r.get("image_path"),
        "image": r.get("image"),
        "question": r["question"],
        # gold = majority-vote representative answer (design doc Stage 1)
        "gold_answer": str(r.get("multiple_choice_answer")
                           or _majority(r.get("answers", []))),
        "choices": None,
    }


# --- okvqa: lmms-lab/OK-VQA (HF mirror is val-only; official train2014
#     annotations use the same field names) --------------------------------
def _map_okvqa(r: Dict, i: int) -> Dict:
    return {
        "id": f"okvqa_{r.get('question_id', i)}",
        "source": "okvqa",
        "image_path": r.get("image_path"),
        "image": r.get("image"),
        "question": r["question"],
        "gold_answer": _majority(r.get("answers", [])),
        "choices": None,
    }


# --- scienceqa: derek-thomas/ScienceQA -------------------------------------
def _map_scienceqa(r: Dict, i: int) -> Dict:
    question = r["question"]
    hint = (r.get("hint") or "").strip()
    if hint:  # design doc Stage 1: merge hint as "Context: ..."
        question = f"{question} Context: {hint}"
    choices = list(r.get("choices") or [])
    ans_idx = r.get("answer")
    gold = (choices[ans_idx]
            if isinstance(ans_idx, int) and 0 <= ans_idx < len(choices)
            else str(ans_idx))
    return {
        "id": f"scienceqa_{r.get('id', i)}",
        "source": "scienceqa",
        "image_path": r.get("image_path"),
        "image": r.get("image"),
        "question": question,
        "gold_answer": str(gold),   # answer CHOICE TEXT (design doc Stage 1)
        "choices": choices or None,
    }


# --- gqa: lmms-lab/GQA testdev_balanced_instructions (config cached
#     locally; raw keys: id, question, answer, fullAnswer, imageId, ...) ----
def _map_gqa(r: Dict, i: int) -> Dict:
    return {
        "id": f"gqa_{r.get('id', i)}",
        "source": "gqa",
        "image_path": r.get("image_path") or r.get("imageId"),
        "image": r.get("image"),
        "question": r["question"],
        "gold_answer": str(r.get("answer", "")),
        "choices": None,
    }


# --- textvqa: lmms-lab/TextVQA ---------------------------------------------
def _map_textvqa(r: Dict, i: int) -> Dict:
    return {
        "id": f"textvqa_{r.get('question_id', i)}",
        "source": "textvqa",
        "image_path": r.get("image_path"),
        "image": r.get("image"),
        "question": r["question"],
        "gold_answer": _majority(r.get("answers", [])),
        "choices": None,
    }


# --- mmmu: MMMU/MMMU (options stored as a python-list string; answer is a
#     letter index into it) --------------------------------------------------
def _map_mmmu(r: Dict, i: int) -> Dict:
    options = r.get("options")
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            options = [options]
    options = [str(o) for o in (options or [])]
    ans = str(r.get("answer", "")).strip()
    letters = "ABCDEFGHIJ"
    if len(ans) == 1 and ans in letters and letters.index(ans) < len(options):
        gold = options[letters.index(ans)]
    else:
        gold = ans
    return {
        "id": f"mmmu_{r.get('id', i)}",
        "source": "mmmu",
        "image_path": None,
        "image": r.get("image_1"),
        "question": r["question"],
        "gold_answer": gold,
        "choices": options or None,
    }


# --- mathvista: AI4Math/MathVista ------------------------------------------
def _map_mathvista(r: Dict, i: int) -> Dict:
    choices = r.get("choices")
    return {
        "id": f"mathvista_{r.get('pid', i)}",
        "source": "mathvista",
        "image_path": r.get("image"),
        "image": r.get("decoded_image"),
        "question": r.get("query") or r["question"],
        "gold_answer": str(r.get("answer", "")),
        "choices": list(choices) if choices else None,
    }


# --- llava_cot_100k: Xkev/LLaVA-CoT-100k -----------------------------------
# conversations = [{"from": "human", "value": q}, {"from": "gpt", "value": a}]
# with the gpt turn tagged <SUMMARY>/<CAPTION>/<REASONING>/<CONCLUSION>.
# Stage mapping onto the design's 3-stage cot:
#   grounding <- CAPTION (what is visible), reasoning <- REASONING,
#   synthesis <- CONCLUSION (also the final answer text).
def _map_llava_cot(r: Dict, i: int) -> Dict:
    convs = r.get("conversations", [])
    human = next((c["value"] for c in convs if c.get("from") == "human"), "")
    gpt = next((c["value"] for c in convs if c.get("from") == "gpt"), "")
    question = human.replace("<image>", "").strip()
    grounding = _extract_tag(gpt, "CAPTION")
    reasoning = _extract_tag(gpt, "REASONING")
    synthesis = _extract_tag(gpt, "CONCLUSION")
    final_answer = synthesis or gpt.strip()
    out = {
        "id": f"llava_cot_100k_{i}",
        "source": "llava_cot_100k",
        "image_path": r.get("image"),
        "image": None,
        "question": question,
        "gold_answer": final_answer,
        "choices": None,
    }
    if grounding or reasoning or synthesis:
        out["cot"] = {"grounding": grounding, "reasoning": reasoning,
                      "synthesis": synthesis}
        out["cot_steps"] = [grounding, reasoning, synthesis]
        out["final_answer"] = final_answer
    return out


# --- instructcot_200k: LOCAL JSONL (design doc Section 3 schema) -----------
# Required per-row fields are validated 1:1 against the frozen schema names.
_ICOT_REQUIRED = ("id", "source", "image_path", "question", "gold_answer",
                  "ais_complexity_score", "instruction_level",
                  "instruction_text", "cot", "final_answer", "qa_scores",
                  "split")
_ICOT_COT_KEYS = ("grounding", "reasoning", "synthesis")
_ICOT_QA_KEYS = ("answer_consistency", "grounding", "coherence")


def _resolve_image_path(image_path: Optional[str], jsonl_dir: str) -> Optional[str]:
    if not image_path:
        return image_path
    if os.path.isfile(image_path):
        return os.path.abspath(image_path)
    for base in (jsonl_dir, os.path.dirname(jsonl_dir)):
        cand = os.path.join(base, image_path)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return image_path


def _load_instructcot_jsonl(data_path: Optional[str], split: Optional[str],
                            limit: Optional[int]) -> List[Dict]:
    if not data_path:
        raise ValueError("source='instructcot_200k' requires data_path "
                         "(a local JSONL file or a directory of *.jsonl "
                         "files). No network is used.")
    if os.path.isdir(data_path):
        files = sorted(
            os.path.join(data_path, f) for f in os.listdir(data_path)
            if f.endswith(".jsonl")
            and os.path.isfile(os.path.join(data_path, f)))
        if not files:
            raise FileNotFoundError(
                f"No *.jsonl files directly inside directory: {data_path}")
    elif os.path.isfile(data_path):
        files = [data_path]
    else:
        raise FileNotFoundError(f"InstructCoT-200K JSONL not found: {data_path}")
    rows: List[Dict] = []
    done = False
    for fp in files:
        jsonl_dir = os.path.dirname(os.path.abspath(fp))
        with open(fp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = [k for k in _ICOT_REQUIRED if k not in row]
                if missing:
                    raise ValueError(
                        f"{fp}:{ln}: missing schema fields {missing} "
                        "(design doc Section 3 -- field names are frozen)")
                for k in _ICOT_COT_KEYS:
                    if k not in row["cot"]:
                        raise ValueError(f"{fp}:{ln}: cot missing '{k}'")
                for k in _ICOT_QA_KEYS:
                    if k not in row["qa_scores"]:
                        raise ValueError(f"{fp}:{ln}: qa_scores missing '{k}'")
                if split and row.get("split") != split:
                    continue
                row["image_path"] = _resolve_image_path(
                    row.get("image_path"), jsonl_dir)
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    done = True
                    break
        if done:
            break
    if not rows:
        raise ValueError(f"No rows loaded from {data_path} (split={split!r})")
    return rows


def _map_instructcot(r: Dict, i: int) -> Dict:
    image_path = r["image_path"]
    # smoke/stub tolerance: dummy paths yield image=None (path preserved)
    image = None
    return {
        "id": r["id"],
        "source": r["source"],
        "image_path": image_path,
        "image": image,
        "question": r["question"],
        "gold_answer": str(r["gold_answer"]),
        "choices": r.get("choices"),
        # -- design doc Section 3 extras (frozen field names) --
        "ais_complexity_score": float(r["ais_complexity_score"]),
        "instruction_level": int(r["instruction_level"]),
        "instruction_text": r["instruction_text"],
        "cot": r["cot"],
        "cot_steps": [r["cot"]["grounding"],     # r_1 (PCC step order fixed,
                      r["cot"]["reasoning"],     # r_2  design doc Section 11)
                      r["cot"]["synthesis"]],    # r_3
        "final_answer": str(r["final_answer"]),
        "qa_scores": r["qa_scores"],
        "split": r["split"],
    }


# --- register everything ----------------------------------------------------
# HF paths/configs/splits below are the ones VERIFIED present in the local
# cache (HF_HOME=/data3/.cache/huggingface, HF_HUB_OFFLINE=1, 2026-07-23).
register_adapter("vqav2", _hf_loader("lmms-lab/VQAv2",
                                     default_split="validation"), _map_vqav2)
register_adapter("okvqa", _hf_loader("lmms-lab/OK-VQA",
                                     default_split="val2014"), _map_okvqa)
register_adapter("scienceqa", _hf_loader("derek-thomas/ScienceQA",
                                         default_split="validation"),
                 _map_scienceqa)
register_adapter("gqa", _hf_loader("lmms-lab/GQA",
                                   config="testdev_balanced_instructions",
                                   default_split="testdev"),
                 _map_gqa)
register_adapter("textvqa", _hf_loader("lmms-lab/TextVQA",
                                       default_split="validation"),
                 _map_textvqa)
register_adapter("mmmu", _hf_loader("MMMU/MMMU", config="Accounting",
                                    default_split="validation"), _map_mmmu)
register_adapter("mathvista", _hf_loader("AI4Math/MathVista",
                                         default_split="testmini"),
                 _map_mathvista)
register_adapter("llava_cot_100k", _hf_loader("Xkev/LLaVA-CoT-100k"),
                 _map_llava_cot)
register_adapter("instructcot_200k", _load_instructcot_jsonl, _map_instructcot)


# ---------------------------------------------------------------------------
# Universal Dataset
# ---------------------------------------------------------------------------

class InstructCoTDataset(Dataset):
    def __init__(self, source: str, data_path: Optional[str] = None,
                 split: Optional[str] = None, limit: Optional[int] = None):
        if source not in FORMAT_ADAPTERS:
            raise KeyError(
                f"Unknown source '{source}'. Registered formats: "
                f"{sorted(FORMAT_ADAPTERS)}")
        self.source = source
        adapter = FORMAT_ADAPTERS[source]
        self._map = adapter["map"]
        self._raw = adapter["load"](data_path, split, limit)

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, i: int) -> Dict:
        return self._map(self._raw[i], i)


def collate_fn(batch: List[Dict]) -> Dict[str, List]:
    keys = []
    for sample in batch:
        for k in sample:
            if k not in keys:
                keys.append(k)
    out: Dict[str, List] = {}
    for k in keys:
        vals = [s.get(k) for s in batch]
        if (all(isinstance(v, torch.Tensor) for v in vals)
                and len({tuple(v.shape) for v in vals}) == 1):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


if __name__ == "__main__":
    # Offline self-check: adapter map functions on synthetic raw records
    # (no network, no `datasets` import).
    fake = {
        "vqav2": {"question_id": 1, "question": "What color?",
                  "multiple_choice_answer": "red",
                  "answers": [{"answer": "red"}] * 10},
        "okvqa": {"question_id": 2, "question": "What sport?",
                  "answers": ["surfing"] * 6 + ["sailing"] * 4},
        "scienceqa": {"id": 3, "question": "Which is north?",
                      "choices": ["Montana", "Texas"], "answer": 0,
                      "hint": "The map shows the US."},
        "gqa": {"id": 4, "question": "Left or right?", "answer": "right"},
        "textvqa": {"question_id": 5, "question": "What brand?",
                    "answers": ["coca cola"] * 10},
        "mmmu": {"id": 6, "question": "Which curve?",
                 "options": "['A curve', 'B curve']", "answer": "A"},
        "mathvista": {"pid": 7, "question": "What is x?", "answer": "4",
                      "choices": ["3", "4"]},
        "llava_cot_100k": {"image": "x.jpg", "conversations": [
            {"from": "human", "value": "<image>\nWhat animal?"},
            {"from": "gpt", "value": "<SUMMARY>s</SUMMARY><CAPTION>a dog"
             "</CAPTION><REASONING>fur and tail</REASONING>"
             "<CONCLUSION>dog</CONCLUSION>"}]},
    }
    for src, raw in fake.items():
        s = FORMAT_ADAPTERS[src]["map"](raw, 0)
        assert s["source"] == src and s["question"] and s["gold_answer"], src
        print(f"[adapter-ok] {src:16s} gold={s['gold_answer']!r} "
              f"choices={s['choices']}")
    print(f"[registry] {len(FORMAT_ADAPTERS)} formats: "
          f"{sorted(FORMAT_ADAPTERS)}")
