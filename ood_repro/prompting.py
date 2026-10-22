"""Auditable few-shot prompt serialization; both chat and plain modes are explicit."""
import json
from importlib.resources import files
from pathlib import Path

from .data import digest

SYSTEM = {
    "csqa": "You are a plain-text reasoning engine. You must strictly follow the reasoning format shown in the examples. Do not start new paragraphs or lines. End each answer with: 'So the answer is (<letter>).'",
    "gsm8k": "You are a mathematical reasoning engine. You must strictly follow the reasoning format shown in the examples. Use plain ASCII text only. Do not start new paragraphs or lines. End each answer with: 'The answer is <number>.'",
    "aqua": "You are a mathematical reasoning engine. You must strictly follow the reasoning format shown in the examples. Use plain ASCII text only. Do not start new paragraphs or lines. End each answer with: 'The answer is (<letter>).'",
}


def load_pack(task, path=None):
    resource = Path(path) if path else files("ood_repro").joinpath("prompts", f"{task}.json")
    pack = json.loads(resource.read_text(encoding="utf-8"))
    if not pack.get("examples") or any(not x.get("question") or not x.get("answer") for x in pack["examples"]):
        raise ValueError("Prompt pack must contain nonempty question/answer examples")
    return pack


def build_prompt(tokenizer, example, task, pack, mode="chat"):
    question = example["question"]
    if example["options"]:
        question += "\nAnswer Choices:\n" + "\n".join(example["options"])
    prefix = "\n\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in pack["examples"])
    content = prefix + f"\n\nQ: {question}\nA:"
    if mode == "chat":
        messages = [{"role": "system", "content": SYSTEM[task]}, {"role": "user", "content": content}]
        if not tokenizer.chat_template:
            raise ValueError("Tokenizer has no chat template; choose --prompt-mode plain explicitly")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        text = SYSTEM[task] + "\n\n" + content
        ids = tokenizer.encode(text, add_special_tokens=True)
    return text, ids, {"prompt_hash": digest(text), "pack_hash": digest(pack), "prompt_mode": mode}
