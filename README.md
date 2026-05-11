# DPO Pipeline

End-to-end Direct Preference Optimization (DPO) pipeline for fine-tuning LLMs on preference data.

## Overview

This pipeline provides a complete workflow for DPO fine-tuning:

1. **Pair Generation** - Generate preference pairs from existing datasets using a weak local model
2. **DPO Training** - Train with LoRA using TRL's DPOTrainer
3. **Merge & Upload** - Merge adapters, test inference, and push to HuggingFace Hub
4. **Eval Tool** - Interactive HTML tool for calibrating preference pairs before training

All scripts use a CONFIG block at the top - no command-line arguments needed.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Generate preference pairs
python dpo_generate_pairs.py

# 2. Train DPO model
python dpo_train.py

# 3. Merge and upload
python dpo_merge_upload.py
```

## Scripts

### `dpo_generate_pairs.py`
Generates rejected responses using a weak local model to create preference pairs.

**What it does:**
- Loads your SFT dataset (from HuggingFace or local file)
- Generates weaker responses using a small local model (LM Studio, Ollama, llama.cpp)
- Filters pairs by quality (length, similarity)
- Outputs JSONL with `prompt`, `chosen`, `rejected` fields

**Configuration:**
- Set your local model endpoint (works with any OpenAI-compatible API)
- Choose a weak model (0.5B-1.5B recommended)
- Adjust quality filters (similarity threshold, min length)
- Supports concurrent requests and retry logic

### `dpo_train.py`
Trains a DPO model using LoRA on the generated preference pairs.

**What it does:**
- Loads base model and tokenizer
- Applies chat templates (Qwen, Llama, or auto-detect)
- Sets up LoRA configuration
- Trains with DPOTrainer from TRL
- Saves adapter locally
- Optional: pushes to HuggingFace Hub

**Configuration:**
- Base model (HF ID or local path)
- Dataset (local JSONL or HF dataset)
- LoRA parameters (r, alpha, target modules)
- DPO hyperparameters (beta, loss type)
- Training settings (epochs, batch size, learning rate)
- HuggingFace upload options

### `dpo_merge_upload.py`
Merges LoRA adapter into base model, runs inference tests, and uploads to Hub.

**What it does:**
- Merges adapter weights into base model
- Runs interactive inference test with sample prompts
- Generates model card README
- Uploads merged model to HuggingFace Hub

**Configuration:**
- Base model and adapter paths
- Upload settings (repo ID, private/public)
- Inference test prompts
- Training metadata for model card

### `dpo_eval_tool.html`
Interactive web-based tool for reviewing and calibrating preference pairs.

**Features:**
- Side-by-side comparison of chosen vs rejected responses
- Quality filtering and statistics
- Export filtered pairs for training
- Dark theme optimized for extended review sessions

**Usage:**
Open directly in browser - no server needed. Load your JSONL pair file and review quality before training.

## Configuration

All scripts use a `CONFIG` dictionary at the top. Edit only this section - no need to modify the code below.

Example from `dpo_train.py`:
```python
CONFIG = {
    "base_model": "your-base-model",
    "dataset_name": "your_pairs.jsonl",
    "lora_r": 16,
    "beta": 0.125,
    "num_train_epochs": 2,
    # ... other settings
}
```

## Recommended Workflow

1. **Generate pairs** with a conservative similarity threshold (0.85)
2. **Review in eval tool** - check distribution, filter outliers
3. **Quick test** - train on 50-100 pairs with `max_samples` to validate setup
4. **Full training** - run on complete dataset
5. **Merge & test** - verify merged model quality before upload
6. **Publish** - push to HuggingFace with auto-generated model card

## Requirements

- Python 3.10+
- CUDA GPU (8GB+ VRAM for 3B models with LoRA)
- Local model server (LM Studio, Ollama, or llama.cpp) for pair generation
- HuggingFace account for uploads (optional)

## Hardware Notes

Tested on:
- RTX 5090 32GB
- 128GB RAM
- Python 3.10, PyTorch 2.x, CUDA 12.x

For smaller GPUs:
- Reduce batch size and gradient accumulation
- Use 1B-3B base models
- Consider fp16 instead of bf16 if needed

## License

MIT

## References

- [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
