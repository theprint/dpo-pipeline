"""
DPO Pair Generator
Loads an existing dataset of good responses, generates weaker responses
using a local model via OpenAI-compatible endpoint, and saves output as
JSONL with chosen/rejected fields ready for DPO training or the eval tool.

All configuration is in the CONFIG section below.
"""

# ============================================================
#  CONFIG — edit this section only
# ============================================================

CONFIG = {
    # --- Input dataset ---
    "input_file": "theprint/Tom-4.2k-alpaca",
    #   JSONL file where each line has at least a prompt and a good response.
    #   Set to a HuggingFace dataset ID to load from HF instead, e.g.:
    #   "input_file": "theprint/my-sft-dataset"
    #   In that case also set input_hf_split below.
    "input_hf": True,                  # True = load from HuggingFace
    "input_hf_split": "train",

    # --- Input field names ---
    #   What are the prompt and good response fields called in your dataset?
    "field_prompt":   "instruction",
    "field_chosen":   "output",       # The existing good response field

    # --- Output ---
    "output_file": "tom_dpo_pairs.jsonl",
    #   Output fields will always be: prompt, chosen, rejected
    #   Compatible with DPOTrainer and the DPO eval tool.

    # --- Sample selection ---
    "max_samples": None,                 # How many pairs to generate. None = all.
    "shuffle": True,                    # Shuffle before selecting max_samples.
    "seed": 42,

    # --- Local model endpoint (OpenAI-compatible) ---
    #   Works with LM Studio, Ollama, llama.cpp server, vLLM, etc.
    "endpoint_url": "http://localhost:1234/v1",
    "weak_model":   "llama-3.2-1b-instruct",
    #   Use a small, weak model — the goal is a noticeably worse response.
    #   Good candidates: 0.5B–1.5B models, untuned base models, or pruned models.
    "api_key": "lm-studio",             # LM Studio accepts any non-empty string.

    # --- Weak model generation settings ---
    #   These are intentionally set to produce lower-quality output.
    "weak_temperature":   1.1,          # Higher than normal — more random
    "weak_max_tokens":    4000,
    "weak_top_p":         0.95,
    #   System prompt for the weak model. Keep it minimal — no CoT instruction,
    #   no persona, no quality guidance. The goal is a generic weak response.
    "weak_system_prompt": "You are a helpful assistant.",

    # --- Quality filter ---
    #   Skip pairs where the weak response is too similar to chosen
    #   or suspiciously short/empty. Set to None to disable.
    "min_rejected_length":  20,         # Minimum chars for rejected response
    "min_length_ratio":     0.1,        # rejected must be at least this fraction
    #                                     of chosen length (catches empty outputs)
    "skip_if_too_similar":  True,       # Skip if responses share too many words
    "similarity_threshold": 0.85,       # Jaccard similarity ceiling (0–1)

    # --- Retry settings ---
    "max_retries":    3,                # Retries per sample on API failure
    "retry_delay":    2.0,              # Seconds between retries
    "request_timeout": 60,             # Seconds before request times out

    # --- Concurrency ---
    "concurrent_requests": 1,          # Keep at 1 for local models unless your
    #                                    server explicitly supports parallel requests.

    # --- Logging ---
    "log_skipped": True,               # Log samples that were filtered out
    "progress_every": 10,              # Print progress every N samples
}

# ============================================================
#  END CONFIG
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

import requests

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================
#  Data loading
# ============================================================

def load_input(cfg: dict) -> list[dict]:
    if cfg["input_hf"]:
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets --break-system-packages")
            sys.exit(1)

        logger.info(f"Loading HuggingFace dataset: {cfg['input_file']} ({cfg['input_hf_split']})")
        ds = load_dataset(cfg["input_file"], split=cfg["input_hf_split"])
        records = [dict(row) for row in ds]
    else:
        path = Path(cfg["input_file"])
        if not path.exists():
            logger.error(f"Input file not found: {path}")
            sys.exit(1)
        logger.info(f"Loading local file: {path}")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        # Detect format: JSON array vs JSONL
        if raw.startswith("["):
            # Standard JSON array
            try:
                records = json.loads(raw)
                if not isinstance(records, list):
                    logger.error("JSON file does not contain a top-level array.")
                    sys.exit(1)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file: {e}")
                sys.exit(1)
        else:
            # JSONL — one object per line
            for i, line in enumerate(raw.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {i+1}: {e}")

    logger.info(f"Loaded {len(records)} records")

    # Validate fields
    fp = cfg["field_prompt"]
    fc = cfg["field_chosen"]
    missing = [r for r in records[:5] if fp not in r or fc not in r]
    if missing:
        sample_keys = list(records[0].keys()) if records else []
        logger.error(
            f"Field mapping mismatch.\n"
            f"  Expected fields: '{fp}', '{fc}'\n"
            f"  Found in dataset: {sample_keys}\n"
            f"  Update field_prompt / field_chosen in CONFIG."
        )
        sys.exit(1)

    # Filter empty
    valid = [r for r in records if r.get(fp, "").strip() and r.get(fc, "").strip()]
    dropped = len(records) - len(valid)
    if dropped:
        logger.warning(f"Dropped {dropped} records with empty prompt or response")

    # Shuffle and limit
    if cfg["shuffle"]:
        random.seed(cfg["seed"])
        random.shuffle(valid)

    if cfg["max_samples"] is not None:
        valid = valid[:cfg["max_samples"]]
        logger.info(f"Using {len(valid)} samples (max_samples={cfg['max_samples']})")

    return valid


# ============================================================
#  Weak model client
# ============================================================

class LocalModelClient:
    def __init__(self, cfg: dict):
        self.base_url  = cfg["endpoint_url"].rstrip("/")
        self.model     = cfg["weak_model"]
        self.api_key   = cfg["api_key"]
        self.timeout   = cfg["request_timeout"]
        self.system    = cfg["weak_system_prompt"]
        self.gen_params = {
            "temperature": cfg["weak_temperature"],
            "max_tokens":  cfg["weak_max_tokens"],
            "top_p":       cfg["weak_top_p"],
        }

    def generate(self, prompt: str) -> Optional[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user",   "content": prompt},
            ],
            **self.gen_params,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            logger.error(
                f"Cannot connect to {self.base_url}. "
                f"Is your local server running? (LM Studio / Ollama / llama.cpp)"
            )
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out after {self.timeout}s")
            return None
        except (KeyError, IndexError) as e:
            logger.warning(f"Unexpected response format: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error: {e}")
            return None

    def check_connection(self) -> bool:
        """Quick connectivity check before starting the main loop."""
        try:
            url = f"{self.base_url}/models"
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            models = [m.get("id","") for m in resp.json().get("data", [])]
            if models:
                logger.info(f"Connected. Available models: {models}")
            else:
                logger.info(f"Connected to {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False


# ============================================================
#  Quality filtering
# ============================================================

def jaccard_similarity(a: str, b: str) -> float:
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def is_acceptable(chosen: str, rejected: str, cfg: dict) -> tuple[bool, str]:
    """Returns (acceptable, reason_if_not)."""

    if not rejected or not rejected.strip():
        return False, "empty response"

    if cfg["min_rejected_length"] and len(rejected.strip()) < cfg["min_rejected_length"]:
        return False, f"too short ({len(rejected.strip())} chars)"

    if cfg["min_length_ratio"]:
        ratio = len(rejected) / max(len(chosen), 1)
        if ratio < cfg["min_length_ratio"]:
            return False, f"length ratio too low ({ratio:.2f})"

    if cfg["skip_if_too_similar"]:
        sim = jaccard_similarity(chosen, rejected)
        if sim > cfg["similarity_threshold"]:
            return False, f"too similar to chosen (Jaccard={sim:.2f})"

    return True, ""


# ============================================================
#  Main generation loop
# ============================================================

def generate_pairs(cfg: dict):
    records = load_input(cfg)
    client  = LocalModelClient(cfg)

    logger.info(f"Checking connection to {cfg['endpoint_url']} ...")
    if not client.check_connection():
        logger.error("Aborting — cannot reach local model server.")
        sys.exit(1)

    output_path = Path(cfg["output_file"])
    fp = cfg["field_prompt"]
    fc = cfg["field_chosen"]

    total     = len(records)
    written   = 0
    skipped   = 0
    failed    = 0

    logger.info(f"Generating {total} rejected responses using: {cfg['weak_model']}")
    logger.info(f"Output → {output_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, record in enumerate(records):

            if i > 0 and i % cfg["progress_every"] == 0:
                logger.info(
                    f"Progress: {i}/{total} | "
                    f"written={written} skipped={skipped} failed={failed}"
                )

            prompt = record[fp].strip()
            chosen = record[fc].strip()

            # Retry loop
            rejected = None
            for attempt in range(cfg["max_retries"]):
                result = client.generate(prompt)
                if result is not None:
                    rejected = result
                    break
                if attempt < cfg["max_retries"] - 1:
                    logger.warning(f"  Retry {attempt+1}/{cfg['max_retries']} for sample {i+1}")
                    time.sleep(cfg["retry_delay"])

            if rejected is None:
                failed += 1
                logger.warning(f"  Sample {i+1}: all retries failed, skipping")
                continue

            # Quality filter
            ok, reason = is_acceptable(chosen, rejected, cfg)
            if not ok:
                skipped += 1
                if cfg["log_skipped"]:
                    logger.info(f"  Sample {i+1}: filtered — {reason}")
                continue

            # Write output
            pair = {
                "prompt":   prompt,
                "chosen":   chosen,
                "rejected": rejected,
            }
            out_f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            written += 1

    # Summary
    logger.info("─" * 50)
    logger.info(f"Complete.")
    logger.info(f"  Total processed : {total}")
    logger.info(f"  Pairs written   : {written}")
    logger.info(f"  Filtered/skipped: {skipped}")
    logger.info(f"  Failed (API)    : {failed}")
    logger.info(f"  Output file     : {output_path.resolve()}")

    if written == 0:
        logger.warning(
            "No pairs were written. Check your endpoint, model name, and field mapping."
        )
    elif written < 50:
        logger.warning(
            f"Only {written} pairs generated. "
            f"You need at least 50 for a reliable calibration session. "
            f"Consider increasing max_samples or relaxing quality filters."
        )
    else:
        logger.info(
            f"Ready for DPO eval tool or DPO training. "
            f"Load {output_path.name} with fields: prompt / chosen / rejected."
        )


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    generate_pairs(CONFIG)
