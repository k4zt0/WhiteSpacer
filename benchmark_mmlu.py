#!/usr/bin/env python3
"""Measure WhiteSpacer and base-model MMLU multiple-choice accuracy."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer


LABELS = ("A", "B", "C", "D")
DEFAULT_SUBJECTS = (
    "computer_security",
    "college_computer_science",
    "high_school_computer_science",
)


def answer_index(answer: Any) -> int:
    if isinstance(answer, str):
        return LABELS.index(answer)
    return int(answer)


def format_question(row: dict[str, Any], include_answer: bool) -> str:
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(LABELS, row["choices"])
    )
    result = f"Question: {row['question']}\n{choices}\nAnswer:"
    if include_answer:
        result += f" {LABELS[answer_index(row['answer'])]}"
    return result


def build_prompt(
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    demonstrations: list[dict[str, Any]],
) -> str:
    examples = "\n\n".join(
        format_question(example, include_answer=True) for example in demonstrations
    )
    content = (
        "Use the examples to answer the final multiple-choice question. "
        "Return only A, B, C, or D.\n\n"
        f"{examples}\n\n{format_question(row, include_answer=False)}"
    )
    return tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are a precise cybersecurity and computer-science analyst. "
                    "Answer multiple-choice questions using only the requested letter."
                ),
            },
            {"role": "user", "content": content},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.inference_mode()
def predict(
    model: AutoPeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> int:
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
    )["input_ids"]
    candidates = [f"{prompt} {label}" for label in LABELS]
    encoded = tokenizer(
        candidates,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    encoded.pop("token_type_ids", None)
    outputs = model(**encoded, use_cache=False)
    log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)
    scores = []
    for index in range(len(LABELS)):
        actual_length = int(encoded["attention_mask"][index].sum())
        candidate_ids = encoded["input_ids"][index].tolist()
        if candidate_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Candidate tokenization does not preserve the prompt prefix")
        score = 0.0
        for position in range(len(prompt_ids), actual_length):
            token_id = encoded["input_ids"][index, position]
            score += float(log_probs[index, position - 1, token_id])
        scores.append(score)
    return max(range(len(scores)), key=scores.__getitem__)


def evaluate_subject(
    model: AutoPeftModelForCausalLM,
    tokenizer: AutoTokenizer,
    subject: str,
    shots: int,
    disable_adapter: bool,
    limit: int | None,
) -> dict[str, Any]:
    demonstrations = list(load_dataset("cais/mmlu", subject, split="dev"))[:shots]
    test = list(load_dataset("cais/mmlu", subject, split="test"))
    if limit is not None:
        test = test[:limit]
    correct = 0
    context = model.disable_adapter() if disable_adapter else nullcontext()
    with context:
        for number, row in enumerate(test, 1):
            prediction = predict(
                model,
                tokenizer,
                build_prompt(tokenizer, row, demonstrations),
            )
            correct += prediction == answer_index(row["answer"])
            if number % 10 == 0 or number == len(test):
                print(
                    f"{subject} {'base' if disable_adapter else 'adapter'}: "
                    f"{number}/{len(test)}",
                    flush=True,
                )
    return {
        "correct": correct,
        "total": len(test),
        "accuracy": correct / len(test),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="KaztoRay/WhiteSpacer")
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mmlu-security.json"),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    model.eval()

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for subject in args.subjects:
        results[subject] = {
            "base": evaluate_subject(
                model,
                tokenizer,
                subject,
                args.shots,
                disable_adapter=True,
                limit=args.limit,
            ),
            "WhiteSpacer": evaluate_subject(
                model,
                tokenizer,
                subject,
                args.shots,
                disable_adapter=False,
                limit=args.limit,
            ),
        }

    totals = {}
    for model_name in ("base", "WhiteSpacer"):
        correct = sum(result[model_name]["correct"] for result in results.values())
        total = sum(result[model_name]["total"] for result in results.values())
        totals[model_name] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }
    report = {
        "model": args.model,
        "method": f"{args.shots}-shot conditional log-likelihood",
        "subjects": results,
        "weighted_total": totals,
        "accuracy_delta": (
            totals["WhiteSpacer"]["accuracy"] - totals["base"]["accuracy"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
