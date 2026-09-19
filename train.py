#!/usr/bin/env python3
"""QLoRA supervised fine-tuning for WhiteSpacer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    required = {"model_name", "dataset_path", "output_dir", "target_modules"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this QLoRA configuration")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(
        config.get("tokenizer_name", config["model_name"]),
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
    )
    model.config.use_cache = False
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=bool(config["gradient_checkpointing"]),
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=int(config["lora_r"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            bias="none",
            target_modules=list(config["target_modules"]),
        ),
    )
    model.print_trainable_parameters()

    files = {"train": config["dataset_path"]}
    eval_path = config.get("eval_dataset_path")
    if eval_path and Path(eval_path).exists():
        files["validation"] = eval_path
    dataset = load_dataset("json", data_files=files)
    max_length = int(config["max_seq_length"])

    def tokenize(row: dict[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError("Each row must end with one assistant response")
        prompt = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            full,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=max_length, add_special_tokens=False
        )["input_ids"]
        labels = list(encoded["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(
            len(prompt_ids), len(labels)
        )
        encoded["labels"] = labels
        return encoded

    tokenized = dataset.map(
        tokenize,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing and masking prompts",
    )
    has_eval = "validation" in tokenized
    training_args = TrainingArguments(
        output_dir=args.output_dir or config["output_dir"],
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        warmup_ratio=float(config["warmup_ratio"]),
        weight_decay=float(config["weight_decay"]),
        logging_steps=int(config["logging_steps"]),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(config["eval_steps"]) if has_eval else None,
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=bool(config["gradient_checkpointing"]),
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        seed=int(config["seed"]),
        max_steps=args.max_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    output_dir = args.output_dir or config["output_dir"]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
