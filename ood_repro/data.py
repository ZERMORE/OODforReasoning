"""Dataset normalization, deterministic train calibration selection, strict extraction."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

SPECS = {
    "csqa": ("tau/commonsense_qa", None, "validation", 1221),
    "gsm8k": ("openai/gsm8k", "main", "test", 1319),
    "aqua": ("deepmind/aqua_rat", "raw", "validation", 254),
    "math": ("EleutherAI/hendrycks_math", None, "test", 5000),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def question_key(text):
    return digest(" ".join(text.split()).casefold())


def normalize(row, task, index):
    if task == "csqa":
        options = [f"({label.lower()}) {text}" for label, text in zip(row["choices"]["label"], row["choices"]["text"])]
        gold = row["answerKey"].upper()
    elif task == "aqua":
        options = [re.sub(r"^([A-E])\s*\)", lambda m: f"({m[1].lower()})", text.strip()) for text in row["options"]]
        gold = row["correct"].upper()
    elif task == "math":
        gold = last_boxed(row["solution"])
        if not gold:
            raise ValueError(f"Missing boxed MATH reference at {index}")
        return {"sample_id": str(row.get("id", f"{task}:{index}")), "question": row["problem"],
                "options": [], "gold_answer": gold, "level": row.get("level"), "subject": row.get("type")}
    else:
        options = []
        gold = number(row["answer"].split("####")[-1].strip())
    if gold is None or (task != "gsm8k" and gold not in list("ABCDE")):
        raise ValueError(f"Missing/invalid gold label at {task}:{index}; unlabeled splits are not supported")
    return {"sample_id": str(row.get("id", f"{task}:{index}")), "question": row["question"], "options": options, "gold_answer": gold}


def load_examples(task, split=None, local_jsonl=None, revision=None, limit=None, seed=0, shuffle=False, exclude_questions=()):
    name, config, default_split, expected = SPECS[task]
    split = split or default_split
    if local_jsonl:
        raw = [json.loads(line) for line in Path(local_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
        examples = [normalize(row, task, f"{split}:{i}") for i, row in enumerate(raw)]
        fingerprint = digest(raw)
    else:
        if task == "math":
            raise ValueError("Use a pinned local MATH JSONL combining all seven subject configurations")
        from datasets import load_dataset
        dataset = load_dataset(name, name=config, split=split, revision=revision)
        if split == default_split and len(dataset) != expected:
            raise ValueError(f"Unexpected {task} split size: {len(dataset)} != {expected}")
        examples = [normalize(row, task, f"{split}:{i}") for i, row in enumerate(dataset)]
        fingerprint = dataset._fingerprint
    excluded = {question_key(text) for text in exclude_questions}
    before_exclusion = len(examples)
    examples = [row for row in examples if question_key(row["question"]) not in excluded]
    excluded_count = before_exclusion - len(examples)
    if shuffle:
        random.Random(seed).shuffle(examples)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        examples = examples[:limit]
    if len({row["sample_id"] for row in examples}) != len(examples):
        raise ValueError("Duplicate sample IDs")
    return examples, {"dataset": name, "config": config, "split": split, "revision_requested": revision,
                      "fingerprint": fingerprint, "local_jsonl": str(local_jsonl) if local_jsonl else None,
                      "sample_ids": [row["sample_id"] for row in examples], "sample_hash": digest(examples),
                      "excluded_demo_questions": excluded_count,
                      "question_hashes": [question_key(row["question"]) for row in examples]}


def number(text):
    try:
        value = Decimal(text.replace(",", "").strip())
        return str(value.normalize()) if value.is_finite() else None
    except InvalidOperation:
        return None


def extract_answer(response, task):
    """Use the last required answer marker, never any incidental reasoning number."""
    if task == "math":
        return last_boxed(response)
    if task == "gsm8k":
        # Optional terminal period, commas and scientific notation are normalized.
        marker = r"The answer is\s+"
        value = r"([+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
        pattern = marker + value + r"\.?\s*$"
    else:
        marker = r"So the answer is" if task == "csqa" else r"The answer is"
        pattern = marker + r"\s*\(([a-e])\)\.?\s*$"
    matches = list(re.finditer(pattern, response.strip(), flags=re.IGNORECASE))
    if not matches:
        return None
    raw = matches[-1][1]
    return number(raw) if task == "gsm8k" else raw.upper()


def last_boxed(text):
    """Read the final boxed answer, including the dataset's unbraced form."""
    markers = list(re.finditer(r"\\(?:boxed|fbox)(?![A-Za-z])\s*", text))
    if not markers:
        return None
    start = markers[-1].end()
    if text[start:start + 1] != "{":
        unbraced = re.match(r"([^$\n]+)\$", text[start:])
        if unbraced:
            return unbraced[1].strip() or None
        return None
    depth = 1
    for index in range(start + 1, len(text)):
        if text[index - 1] == "\\":
            continue
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start + 1:index].strip() or None
    return None


def grade_math(parsed, gold):
    if parsed is None:
        return False, "no_boxed_answer"
    if parsed == gold:
        return True, "exact"
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException
    config = [LatexExtractionConfig()]
    # Math-Verify 0.8's Windows timeout wrapper cannot pickle its worker.
    timeout = 5 if os.name == "posix" else None
    try:
        expected = parse(f"${gold}$", extraction_config=config, fallback_mode="no_fallback",
                         parsing_timeout=timeout, raise_on_error=True)
        actual = parse(f"${parsed}$", extraction_config=config, fallback_mode="no_fallback",
                       parsing_timeout=timeout, raise_on_error=True)
        if not expected:
            return False, "reference_parse_failure"
        if not actual:
            return False, "prediction_parse_failure"
        return bool(verify(expected, actual, timeout_seconds=timeout, raise_on_error=True)), "symbolic"
    except TimeoutException:
        return False, "grader_timeout"
    except (ValueError, TypeError, ArithmeticError, RuntimeError):
        return False, "grader_error"
