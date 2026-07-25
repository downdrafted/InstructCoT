# InstructCoT

Implementation for:

> **InstructCoT: Bridging Instruction Tuning and Chain-of-Thought Reasoning for Scalable Vision-Language Models**
> Eunsung You, Sangyup Oh, IEEE BigDataService 2026

InstructCoT combines three modules on top of an existing vision-language model (VLM):

- **AIS** (Adaptive Instruction Selector): estimates per-sample difficulty from predictive entropy and routes each example to one of three instruction-detail tiers.
- **ICA** (Instruction-Conditioned Attention): a lightweight, gated cross-attention block that conditions visual features directly on the selected instruction.
- **PCC** (Progressive CoT Curriculum): a three-stage training schedule that gradually increases reasoning supervision from direct answers → short rationales → full grounding/reasoning/synthesis chain-of-thought.

This repo contains the three modules, the data pipeline, and both a CPU "stub" path (for verifying the wiring with no GPU) and a real path (LLaVA-1.5 via `transformers`, 4-bit quantized).

## Repository structure

| File | What it does |
|---|---|
| `ais.py` | Predictive-entropy scoring, threshold calibration (28th/72nd percentile, fixed once), instruction-level routing |
| `ica.py` | The cross-attention module: visual features as query, instruction embedding as key/value, zero-initialized gated residual |
| `pcc.py` | The three-phase curriculum scheduler and the composite training loss |
| `dataset.py` | Adapters that normalize VQAv2, OK-VQA, ScienceQA, GQA, TextVQA, MMMU, MathVista, LLaVA-CoT-100K, and a local InstructCoT-200K JSONL into one common sample format |
| `real_vlm.py` | Wires AIS/ICA/PCC onto an actual LLaVA-1.5 checkpoint in 4-bit via `transformers` + `bitsandbytes` |
| `train_instructcot.py` | Training loop: AIS routes → ICA conditions → PCC selects supervision mode → composite loss → backward |
| `infer_instructcot.py` | Inference: AIS routing + entropy-gated CoT-skipping at generation time |
| `sample_data/sample.jsonl` | A 5-row example in the exact InstructCoT-200K schema, for a quick end-to-end smoke test with no external data or network access |

Every file with training/inference logic has a **stub path** (a tiny synthetic CPU model, for verifying that AIS → ICA → PCC → loss → backward all run correctly) and a **real path** (an actual VLM on GPU). Stub runs print a `[smoke] ... PASSED` line ending in *"plumbing checks, not results"*: passing a stub smoke test confirms the code runs, not that it reproduces any accuracy number from the paper.

## Installation

```bash
# Core (enough for the CPU stub path)
pip install torch numpy

# Additional, for the real GPU path
pip install transformers bitsandbytes peft accelerate pillow
# Optional, only needed to load the public benchmarks (VQAv2, OK-VQA, etc.)
# rather than a local instructcot_200k JSONL:
pip install datasets
```

The real path requires a CUDA GPU. `real_vlm.py` raises immediately if none is available.

## Quickstart

Module-level checks; each file has its own self-check, no data required:

```bash
python3 ais.py
python3 ica.py
python3 pcc.py
```

Full stub pipeline, using the bundled sample data:

```bash
python3 train_instructcot.py --stub --smoke \
    --source instructcot_200k --data sample_data/sample.jsonl --steps 3

python3 infer_instructcot.py --stub --smoke \
    --source instructcot_200k --data sample_data/sample.jsonl --max-samples 5
```

## Data format (InstructCoT-200K schema)

To use your own data with `--source instructcot_200k`, each line of your JSONL must be a JSON object with these fields (validated on load — a row missing any of these is rejected with the exact field name):

| Field | Type | Notes |
|---|---|---|
| `id` | str | Unique row identifier |
| `source` | str | Origin dataset tag |
| `image_path` | str | Path to the image (resolved relative to the JSONL's directory if not absolute) |
| `question` | str | The visual question |
| `gold_answer` | str | Reference answer |
| `ais_complexity_score` | float | Precomputed entropy score, if you're supplying your own rather than letting AIS compute it |
| `instruction_level` | int | 1 / 2 / 3 |
| `instruction_text` | str | The instruction actually shown to the model |
| `cot` | object | `{"grounding": ..., "reasoning": ..., "synthesis": ...}` |
| `final_answer` | str | Final answer text (may differ from `gold_answer` if reformatted) |
| `qa_scores` | object | `{"answer_consistency": ..., "grounding": ..., "coherence": ...}` |
| `split` | str | e.g. `"train"` / `"val"` |

See `sample_data/sample.jsonl` for five complete example rows.

## Running on a real VLM

```bash
python3 train_instructcot.py \
    --source instructcot_200k --data /path/to/your/instructcot_200k/ \
    --model llava-hf/llava-1.5-7b-hf --lora --save-ica checkpoints/ica.pt

python3 infer_instructcot.py \
    --source vqav2 --model llava-hf/llava-1.5-7b-hf --ica-ckpt checkpoints/ica.pt
```

Set `HF_HOME` and, if working offline from a pre-downloaded cache, `HF_HUB_OFFLINE=1` for your own environment before running either command.

## Citation

```bibtex
@inproceedings{you2026instructcot,
  title     = {InstructCoT: Bridging Instruction Tuning and Chain-of-Thought Reasoning for Scalable Vision-Language Models},
  author    = {You, Eunsung and Oh, Sangyup},
  booktitle = {IEEE BigDataService 2026},
  year      = {2026},
  address   = {Fukuoka, Japan}
}
```

## License

The source code provided at https://github.com/downdrafted/InstructCoT is licensed under the MIT License. Code and documentation copyright 2026 Eunsung You and Sangyup Oh.
