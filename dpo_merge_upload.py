import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    # Base model used during DPO training
    "base_model": "theprint/GeneralChat-Llama3.2-3B",

    # Path to the DPO output directory (contains adapter_model.safetensors)
    "adapter_path": "./dpo_output",

    # Where to save the merged model locally
    "output_dir": "./dpo_merged",

    # Upload to Hugging Face Hub
    "push_to_hub": True,
    "skip_to_upload": True,  # set True to skip merge and inference, just upload output_dir
    "hub_model_id": "theprint/GeneralChat-Llama3.2-3B-DPO",
    "hub_private": False,

    # dtype for merging — bf16 recommended for Llama
    "torch_dtype": "bfloat16",

    # Inference test — runs before upload, set to False to skip
    "run_inference_test": True,
    "inference_device": "cuda",  # "cuda" or "cpu"
    "inference_max_new_tokens": 256,

    # Training metadata for README
    "training_dataset": "theprint/Tom-4.2k-alpaca",
    "base_model_description": "GeneralChat-Llama3.2-3B, a general-purpose conversational fine-tune of Llama 3.2 3B.",
    "dpo_beta": 0.125,
    "lora_r": 16,
    "lora_alpha": 32,
    "learning_rate": "1e-5",
    "epochs": 3,
    "run_name": "llama3.2-3b-datom-dpo-0310",
}
# ============================================================

README_TEMPLATE = """---
base_model: {base_model}
library_name: transformers
model_name: {model_name}
tags:
- trl
- dpo
- llama
- conversational
license: apache-2.0
---

# {model_name}

A DPO fine-tuned version of [{base_model}](https://huggingface.co/{base_model}).

## Description

{base_model_description}

This model was trained with Direct Preference Optimization (DPO) on the [{training_dataset}](https://huggingface.co/datasets/{training_dataset}) dataset.
Rejected responses were generated using a weak local model to create preference pairs,
with chosen responses drawn from the original dataset.

## Quick Start

```python
from transformers import pipeline

generator = pipeline("text-generation", model="{hub_model_id}", device="cuda")
output = generator(
    [{{"role": "user", "content": "Your prompt here"}}],
    max_new_tokens=256,
    return_full_text=False
)[0]
print(output["generated_text"])
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Method | DPO |
| Base model | {base_model} |
| Dataset | {training_dataset} |
| Beta | {dpo_beta} |
| LoRA r / alpha | {lora_r} / {lora_alpha} |
| Learning rate | {learning_rate} |
| Epochs | {epochs} |
| Run name | {run_name} |

### Framework Versions

- TRL: 0.29.0
- Transformers: 5.3.0
- PyTorch: 2.12.0.dev20260310+cu128
- Datasets: 4.5.0
- PEFT: 0.15.2

## Citation

```bibtex
@inproceedings{{rafailov2023direct,
    title        = {{{{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}}}},
    author       = {{Rafael Rafailov and Archit Sharma and Eric Mitchell and Christopher D. Manning and Stefano Ermon and Chelsea Finn}},
    year         = 2023,
    booktitle    = {{Advances in Neural Information Processing Systems 36}},
    url          = {{http://papers.nips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html}},
}}
```
"""

TEST_PROMPTS = [
    "What's the best way to start learning guitar as a complete beginner?",
    "I've been feeling really unmotivated lately. Any advice?",
    "Explain what a neural network is like I'm twelve.",
    "What should I make for dinner if I only have eggs, cheese, and leftover rice?",
]


def merge_model(cfg, torch_dtype):
    print(f"\n[1/4] Loading base model: {cfg['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        torch_dtype=torch_dtype,
        device_map="cpu",
    )

    print(f"[1/4] Loading adapter from: {cfg['adapter_path']}")
    model = PeftModel.from_pretrained(model, cfg["adapter_path"])

    print("[1/4] Merging adapter into base model...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(cfg["adapter_path"])

    print(f"[1/4] Saving merged model to: {cfg['output_dir']}")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    model.save_pretrained(cfg["output_dir"], safe_serialization=True)
    tokenizer.save_pretrained(cfg["output_dir"])
    print("[1/4] Merge complete.\n")

    return model, tokenizer


def run_inference_test(cfg, model, tokenizer):
    print("[2/4] Running inference test...")
    print(f"      Device: {cfg['inference_device']}")
    print("      Press Enter to use suggested prompt, type your own, or type 'done' to finish.\n")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if cfg["inference_device"] == "cuda" else -1,
        torch_dtype=torch.bfloat16 if cfg["inference_device"] == "cuda" else torch.float32,
    )

    prompt_idx = 0

    while True:
        if prompt_idx < len(TEST_PROMPTS):
            suggestion = f" (suggestion: \"{TEST_PROMPTS[prompt_idx]}\")"
        else:
            suggestion = ""

        user_input = input(f"Prompt{suggestion}\n> ").strip()

        if user_input.lower() == "done":
            break

        if user_input == "" and prompt_idx < len(TEST_PROMPTS):
            user_input = TEST_PROMPTS[prompt_idx]
            print(f"Using: {user_input}")

        prompt_idx += 1

        output = pipe(
            [{"role": "user", "content": user_input}],
            max_new_tokens=cfg["inference_max_new_tokens"],
            return_full_text=False,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )[0]["generated_text"]

        print(f"\nModel: {output}\n")
        print("-" * 60)

    proceed = input("\nLooks good? Proceed to upload? (yes/no): ").strip().lower()
    return proceed in ("yes", "y")


def write_readme(cfg):
    print("[3/4] Generating README...")
    model_name = cfg["hub_model_id"].split("/")[-1]
    readme = README_TEMPLATE.format(
        base_model=cfg["base_model"],
        model_name=model_name,
        hub_model_id=cfg["hub_model_id"],
        base_model_description=cfg["base_model_description"],
        training_dataset=cfg["training_dataset"],
        dpo_beta=cfg["dpo_beta"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        learning_rate=cfg["learning_rate"],
        epochs=cfg["epochs"],
        run_name=cfg["run_name"],
    )
    readme_path = os.path.join(cfg["output_dir"], "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"[3/4] README written to {readme_path}\n")


def upload_to_hub(cfg):
    print(f"[4/4] Uploading to Hub: {cfg['hub_model_id']}")
    
    from huggingface_hub import HfApi
    api = HfApi()
    
    # Create the repo if it doesn't exist
    api.create_repo(
        repo_id=cfg["hub_model_id"],
        private=cfg["hub_private"],
        exist_ok=True,
    )
    
    # Upload the entire merged folder directly
    api.upload_folder(
        folder_path=cfg["output_dir"],
        repo_id=cfg["hub_model_id"],
        repo_type="model",
    )
    
    print(f"[4/4] Done: https://huggingface.co/{cfg['hub_model_id']}\n")


def main():
    cfg = CONFIG

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg["torch_dtype"], torch.bfloat16)

    if cfg.get("skip_to_upload"):
        print("Skipping merge and inference — uploading existing output_dir.")
        write_readme(cfg)
        if cfg["push_to_hub"]:
            upload_to_hub(cfg)  # no model/tokenizer args needed since we're using upload_folder
        return

    # Step 1: Merge
    model, tokenizer = merge_model(cfg, torch_dtype)

    # Step 2: Inference test
    if cfg["run_inference_test"]:
        proceed = run_inference_test(cfg, model, tokenizer)
        if not proceed:
            print("Upload cancelled. Merged model saved locally at:", cfg["output_dir"])
            return

    # Step 3: README
    write_readme(cfg)

    # Step 4: Upload
    if cfg["push_to_hub"]:
        upload_to_hub(cfg)
    else:
        print("push_to_hub is False — skipping upload. Merged model at:", cfg["output_dir"])


if __name__ == "__main__":
    main()
