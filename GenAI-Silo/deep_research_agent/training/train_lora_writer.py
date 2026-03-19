"""
LoRA fine-tuning script for the writer model.

This trains a local HF causal LM to map:
  input_text (query + nuggets + refs prompt) -> output_text (related work with citations)

Usage (example):

  python -m deep_research_agent.training.train_lora_writer \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --train_jsonl path/to/writer_train.jsonl \
    --output_dir path/to/lora_writer

After training, set in `LLMConfig`:
  backend="hf_local"
  hf_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
  hf_lora_path="path/to/lora_writer"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json

from datasets import load_dataset
# Added DataCollatorForLanguageModeling to handle variable sequence lengths
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


@dataclass
class TrainConfig:
    model_name_or_path: str
    train_jsonl: Path
    output_dir: Path
    lr: float = 2e-4
    batch_size: int = 2
    num_epochs: int = 1
    max_length: int = 2048


def load_jsonl_dataset(path: Path):
    """
    Load JSONL with keys: input, output.
    Returns a HF datasets.Dataset.
    """
    return load_dataset("json", data_files=str(path))["train"]


def make_supervised_examples(example, tokenizer, max_length: int):
    """
    Turn {input, output} into a single prompt for causal LM training.
    """
    text = f"<s>[INSTRUCTION]\n{example['input']}\n\n[RESPONSE]\n{example['output']}</s>"
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,  # Let the collator handle padding
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train_lora_writer(cfg: TrainConfig) -> None:
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
    # Ensure a padding token is defined for the collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {cfg.model_name_or_path}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
    )
    
    # Enable gradient checkpointing to save memory
    base_model.gradient_checkpointing_enable()

    # Configure LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)

    dataset = load_jsonl_dataset(cfg.train_jsonl)
    tokenized = dataset.map(
        lambda ex: make_supervised_examples(ex, tokenizer, cfg.max_length),
        remove_columns=list(dataset.column_names),
    )

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        logging_steps=1,
        save_steps=500,
        save_total_limit=2,
        bf16=False,
        fp16=False,  # Disable fp16 on CPU/MPS
        report_to=[],
        remove_unused_columns=False,  # Important for PEFT
        dataloader_pin_memory=False,  # Disable pin_memory on CPU/MPS
        gradient_accumulation_steps=1,
        max_steps=10,  # Limit steps for testing
    )

    # Initialize data collator for causal language modeling with padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=None,  # Pad to max length in batch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator, # Apply padding to avoid ValueError
        processing_class=tokenizer,
    )
    trainer.train()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LoRA fine-tuning for writer model.")
    p.add_argument("--model", required=True, help="Base HF model name or path")
    p.add_argument("--train_jsonl", required=True, help="Writer dataset JSONL (input/output)")
    p.add_argument("--output_dir", required=True, help="Where to save LoRA adapter")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_length", type=int, default=2048)
    args = p.parse_args()

    cfg = TrainConfig(
        model_name_or_path=args.model,
        train_jsonl=Path(args.train_jsonl),
        output_dir=Path(args.output_dir),
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_length=args.max_length,
    )
    train_lora_writer(cfg)
    print(f"Saved LoRA writer adapter to {cfg.output_dir}")