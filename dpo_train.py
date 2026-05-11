"""
DPO Training Script
Supports Qwen and Llama chat templates, configurable datasets, LoRA, and HuggingFace upload.
All configuration is in the CONFIG section below.
"""

# ============================================================
#  CONFIG — edit this section only
# ============================================================

CONFIG = {
    # --- Model ---
    "base_model": "theprint/GeneralChat-Llama3.2-3B",   # HF model ID or local path
    "model_family": "llama",                        # "qwen" | "llama" | "auto"
    #   "auto" will attempt to detect from the tokenizer's chat_template.
    #   Explicit values are safer.

    # --- Dataset ---
    "dataset_name": "tom_dpo_pairs.jsonl",    # HF dataset ID or local path
    "dataset_split": "train",
    #   Field names in your dataset. Adjust to match your actual column names.
    "field_prompt":   "prompt",
    "field_chosen":   "chosen",
    "field_rejected": "rejected",
    #   Optional: limit dataset size for quick tests (set to None to use all)
    "max_samples": None,

    # --- Output ---
    "output_dir": "./dpo_output",
    "run_name": "llama3.2-3b-datom-dpo-0310",

    # --- LoRA ---
    "lora_r": 16,
    "lora_alpha": 32,                              # Typically 2x lora_r
    "lora_dropout": 0.05,
    "lora_target_modules": "all-linear",           # "all-linear" works for most models.
    #   For explicit control use a list: ["q_proj", "v_proj", "k_proj", "o_proj"]

    # --- DPO ---
    "beta": 0.125,                                 # KL penalty. Higher = more conservative.
    #   0.05-0.1: aggressive preference learning
    #   0.1-0.2: balanced (recommended starting point)
    #   0.3+: very conservative, minimal drift from reference
    "loss_type": "sigmoid",                        # "sigmoid" (standard DPO) | "ipo" | "hinge"
    "max_prompt_length": 512,
    "max_length": 4200,                            # prompt + completion combined

    # --- Training ---
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,             # Effective batch = batch_size * grad_accum
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "bf16": True,                                  # Set False if your GPU doesn't support bf16
    "fp16": False,
    "gradient_checkpointing": True,
    "dataloader_num_workers": 4,

    # --- Logging & Saving ---
    "logging_steps": 10,
    "save_steps": 100,
    "save_total_limit": 2,
    "eval_split": 0.05,                           # Fraction of data held out for eval.
    #   Set to 0 to disable eval.

    # --- HuggingFace Upload ---
    "push_to_hub": False,
    "hub_model_id": None,                          # e.g. "theprint/my-dpo-model"
    #   If None, defaults to "{username}/{run_name}"
    "hub_private": False,
    "merge_before_push": True,                    # If True, merges LoRA into base and pushes
    #   full model. If False, pushes adapter only.
    #   Merging requires more VRAM.

    # --- Misc ---
    "seed": 18072005,
    "report_to": "none",                           # "none" | "wandb" | "tensorboard"
}

# ============================================================
#  END CONFIG
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Suppress OpenMP duplicate lib warning on Windows

import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import DPOTrainer, DPOConfig

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
#  Chat template helpers
# ============================================================

SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant."

def apply_qwen_template(tokenizer, prompt: str, system: str = SYSTEM_PROMPT_DEFAULT) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def apply_llama_template(tokenizer, prompt: str, system: str = SYSTEM_PROMPT_DEFAULT) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def detect_model_family(tokenizer) -> str:
    """Best-effort detection from tokenizer chat_template string."""
    ct = getattr(tokenizer, "chat_template", "") or ""
    ct_lower = ct.lower()
    if "qwen" in ct_lower or "<|im_start|>" in ct:
        return "qwen"
    if "llama" in ct_lower or "[inst]" in ct or "<<sys>>" in ct:
        return "llama"
    logger.warning(
        "Could not detect model family from chat template. "
        "Falling back to generic apply_chat_template. "
        "Set model_family explicitly in CONFIG to avoid this."
    )
    return "generic"


def format_prompt(tokenizer, prompt: str, model_family: str) -> str:
    if model_family == "qwen":
        return apply_qwen_template(tokenizer, prompt)
    elif model_family == "llama":
        return apply_llama_template(tokenizer, prompt)
    else:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_DEFAULT},
                {"role": "user", "content": prompt},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt


# ============================================================
#  Dataset preparation
# ============================================================

def load_and_prepare_dataset(cfg: dict, tokenizer) -> DatasetDict:
    path = cfg["dataset_name"]

    # Detect local file vs HuggingFace Hub ID
    if path.endswith(".jsonl") or path.endswith(".json") or os.path.exists(path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Local dataset file not found: {p.resolve()}\n"
                f"Make sure the file is in the same directory as the script, "
                f"or provide a full path."
            )
        logger.info(f"Loading local file: {p.resolve()}")
        ds = load_dataset("json", data_files=str(p), split="train")
    else:
        logger.info(f"Loading from HuggingFace Hub: {path}")
        ds = load_dataset(path, split=cfg["dataset_split"])

    if cfg["max_samples"] is not None:
        ds = ds.select(range(min(cfg["max_samples"], len(ds))))
        logger.info(f"Truncated dataset to {len(ds)} samples")

    # Validate required fields
    missing = [
        f for f in [cfg["field_prompt"], cfg["field_chosen"], cfg["field_rejected"]]
        if f not in ds.column_names
    ]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}\n"
            f"Available columns: {ds.column_names}\n"
            f"Update field_prompt / field_chosen / field_rejected in CONFIG."
        )

    model_family = cfg["model_family"]
    if model_family == "auto":
        model_family = detect_model_family(tokenizer)
        cfg["_resolved_model_family"] = model_family

    def preprocess(example):
        prompt_text = example[cfg["field_prompt"]]
        formatted = format_prompt(tokenizer, prompt_text, model_family)
        return {
            "prompt":   formatted,
            "chosen":   example[cfg["field_chosen"]],
            "rejected": example[cfg["field_rejected"]],
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    # Train / eval split
    if cfg["eval_split"] and cfg["eval_split"] > 0:
        split = ds.train_test_split(test_size=cfg["eval_split"], seed=cfg["seed"])
        logger.info(
            f"Dataset split - train: {len(split['train'])}, eval: {len(split['test'])}"
        )
        return DatasetDict({"train": split["train"], "eval": split["test"]})
    else:
        return DatasetDict({"train": ds})


# ============================================================
#  Model + LoRA setup
# ============================================================

def load_model_and_tokenizer(cfg: dict):
    logger.info(f"Loading base model: {cfg['base_model']}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch.bfloat16 if cfg["bf16"] else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # Required for gradient checkpointing

    return model, tokenizer


def build_lora_config(cfg: dict) -> LoraConfig:
    target = cfg["lora_target_modules"]
    if isinstance(target, str) and target != "all-linear":
        target = [t.strip() for t in target.split(",")]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=target,
        bias="none",
    )


# ============================================================
#  HuggingFace upload
# ============================================================

def push_model(cfg: dict, trainer: DPOTrainer, tokenizer):
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]
    repo_id = cfg["hub_model_id"] or f"{user}/{cfg['run_name']}"

    logger.info(f"Pushing to HuggingFace Hub: {repo_id} (private={cfg['hub_private']})")

    if cfg["merge_before_push"]:
        logger.info("Merging LoRA adapter into base model before push...")
        merged = trainer.model.merge_and_unload()
        merged.push_to_hub(repo_id, private=cfg["hub_private"])
        tokenizer.push_to_hub(repo_id, private=cfg["hub_private"])
        logger.info(f"Merged model pushed to: https://huggingface.co/{repo_id}")
    else:
        trainer.model.push_to_hub(repo_id, private=cfg["hub_private"])
        tokenizer.push_to_hub(repo_id, private=cfg["hub_private"])
        logger.info(f"LoRA adapter pushed to: https://huggingface.co/{repo_id}")


# ============================================================
#  Main
# ============================================================

def main():
    cfg = CONFIG
    set_seed(cfg["seed"])

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Resolve model family if auto
    if cfg["model_family"] == "auto":
        cfg["model_family"] = detect_model_family(tokenizer)
    logger.info(f"Model family: {cfg['model_family']}")

    # Load dataset
    datasets = load_and_prepare_dataset(cfg, tokenizer)

    # LoRA config
    lora_config = build_lora_config(cfg)

    # DPO training config
    # Convert warmup_ratio to warmup_steps
    total_steps = (len(datasets["train"]) // (cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"])) * cfg["num_train_epochs"]
    warmup_steps = max(1, int(total_steps * cfg["warmup_ratio"]))
    logger.info(f"Total steps: {total_steps}, warmup steps: {warmup_steps}")

    training_args = DPOConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg["run_name"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        bf16=cfg["bf16"],
        fp16=cfg["fp16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        save_total_limit=cfg["save_total_limit"],
        report_to=cfg["report_to"],
        seed=cfg["seed"],
        dataloader_num_workers=cfg["dataloader_num_workers"],
        # DPO-specific
        beta=cfg["beta"],
        loss_type=cfg["loss_type"],
        max_length=cfg["max_length"],
        # max_prompt_length=cfg["max_prompt_length"],
    )

    # DPOTrainer handles the frozen reference model internally when peft_config is passed.
    # The base model becomes the reference; the LoRA adapter is the trainable policy.
    eval_dataset = datasets.get("eval", None)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # None + peft_config = auto reference from base weights
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Train
    logger.info("Starting DPO training...")
    trainer.train()

    # Save locally
    logger.info(f"Saving adapter to {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # Eval summary
    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        metrics = trainer.evaluate()
        logger.info(f"Eval metrics: {metrics}")

    # Push to Hub
    if cfg["push_to_hub"]:
        push_model(cfg, trainer, tokenizer)

    logger.info("Done.")


if __name__ == "__main__":
    main()
