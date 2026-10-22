"""Extract only few-shot prefixes from the original authors' public output archive.

Usage: python scripts/import_wei_prompts.py --archive wei-prompts.zip
No execution or blind ZIP extraction. Archive path/content hashes are recorded.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "ood_repro/prompts")
    args = parser.parse_args()
    checksum = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for task, key, expected in [("csqa", "commonsenseqa", 7), ("gsm8k", "gsm", 8), ("aqua", "aqua", 4)]:
            member = f"chain-of-thought-zip/gpt-3-text-davinci-002/{key}_stream/{key}_stream_inputs"
            # Actual newlines separate dataset cases; escaped newlines are inside prompts.
            first_case = archive.read(member).decode("utf-8").splitlines()[0].replace("\\n", "\n")
            prefix = first_case.rsplit("\n\nQ:", 1)[0]
            examples = []
            for text in prefix.split("\n\nQ:"):
                text = text.removeprefix("Q:").strip()
                question, answer = text.split("\nA:", 1)
                examples.append({"question": question, "answer": answer.strip()})
            if len(examples) != expected:
                raise ValueError(f"Unexpected example count: {task} {len(examples)} != {expected}")
            notes = ["Extracted from original Wei et al. authors' GPT-3 output archive, not Liu et al. code.",
                     "Archive order and text retained, including source typos. Exact Liu et al. serialization is unknown."]
            if task == "csqa":
                notes.append("Source fox exemplar says natural habitat but ends (b), although option is (c). Preserved, not silently corrected.")
            if task == "aqua":
                notes.append("Source archive uses 'So the answer is'; Liu appendix asks for 'The answer is'. System instruction follows Liu; examples retain source text.")
            pack = {"source": "https://github.com/jasonwei20/chain-of-thought-prompting",
                    "archive_url": "https://raw.githubusercontent.com/jasonwei20/chain-of-thought-prompting/main/chain-of-thought-zip.zip",
                    "archive_sha256": checksum, "archive_member": member, "notes": notes, "examples": examples}
            target = args.output / f"{task}.json"
            target.write_text(json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{task}: {len(examples)} examples -> {target}")


if __name__ == "__main__":
    main()
