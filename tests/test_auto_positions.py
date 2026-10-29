import json
import sys

import pytest

from ood_repro import auto_positions as module


def make_row(text="We have 2. Then 2 + 2 = 5. The answer is 5.", name="sample", correct=False):
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {b: chr(b) for b in bs}
    for i, b in enumerate(b for b in range(256) if b not in bs):
        mapping[b] = chr(256+i)
    raw = text.encode("utf-8")
    return {"sample_id": name, "source_row_hash": name + "-hash", "is_correct": correct,
            "question": "What is 2 + 2?", "gold_answer": "4", "response": text,
            "generated_token_ids": list(raw), "token_pieces": [mapping[b] for b in raw],
            "sequence_tokens": [{"position": i, "token_id": b, "scores": {"nci": 42}} for i, b in enumerate(raw)]}


def verdict(**updates):
    return {"process_valid": False, "segment_id": 2, "error_quote": "2 + 2 = 5",
            "error_type": "arithmetic", "recovered": False, "notes": "2+2 should equal 4.", **updates}


def test_exact_utf8_alignment_and_quote_boundaries():
    row = make_row("已知2。\n于是2 + 2 = 5。")
    result = module.annotation(row, verdict(error_quote="2 + 2 = 5"), module.byte_alignment(row), "judge")
    assert result["first_error_position"] == len("已知2。\n于是".encode())
    assert row["response"][result["char_start"]:result["char_end"]] == "2 + 2 = 5"
    assert result["human_reviewed"] is False
    row["response"] += "modified"
    with pytest.raises(ValueError, match="differs"):
        module.byte_alignment(row)


def test_ambiguous_quote_and_invalid_judgments_are_rejected():
    row = make_row("2 + 2 = 5; 2 + 2 = 5")
    spans = module.byte_alignment(row)
    with pytest.raises(ValueError, match="ambiguous"):
        module.annotation(row, verdict(segment_id=1), spans, "judge")
    with pytest.raises(ValueError, match="requires"):
        module.annotation(row, verdict(process_valid=True, segment_id=1), spans, "judge")
    result = module.annotation(row, verdict(process_valid=None, segment_id=None, error_quote=None), spans, "judge")
    assert result["first_error_position"] is None and result["process_valid"] is None


def test_payload_is_score_blind_and_segment_ids_contiguous():
    row = make_row("\n\nStart.\nNext 1.5 units.\n")
    parts = module.segments(row["response"])
    assert [s["segment_id"] for s in parts] == [1, 2]
    body = module.payload(row, "deepseek-v4-pro", 8192, "enabled")
    data = json.loads(body["messages"][1]["content"])
    assert set(data) == {"question", "reference_answer", "candidate_response", "segments"}
    assert "sequence_tokens" not in data and "is_correct" not in data


def test_dry_run_limit_resume_and_protocol_guard(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "predictions.jsonl").write_text("source-data")
    output = tmp_path / "judge"
    rows = [make_row(name=str(i), correct=i >= 2) for i in range(4)]
    monkeypatch.setattr(module, "load_replay", lambda _: ({}, rows))
    args = ["judge", "--test", str(source), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args + ["--dry-run"])
    module.main()
    assert not json.loads((output / "judge-manifest.json").read_text())["complete"]
    assert not list((output / "raw").iterdir())
    calls = []

    def api(body, key, base, timeout):
        calls.append(body)
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(verdict())}}],
                "model": "deepseek-v4-pro", "usage": {"total_tokens": 100}}

    monkeypatch.setattr(module, "call_api", api)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "not-a-real-key")
    monkeypatch.setattr(sys, "argv", args + ["--limit", "1"])
    module.main()
    assert len(calls) == 1
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    manifest = json.loads((output / "judge-manifest.json").read_text())
    assert manifest["complete"] and manifest["completed_n"] == 4 and len(calls) == 4
    assert manifest["usage"]["total_tokens"] == 400
    module.main()  # Complete resume must not call the API again.
    assert len(calls) == 4
    assert "not-a-real-key" not in "".join(p.read_text() for p in output.rglob("*.json"))
    monkeypatch.setattr(sys, "argv", args + ["--thinking", "disabled"])
    with pytest.raises(SystemExit):
        module.main()


def test_experiment1_rejects_incomplete_judge_manifest(tmp_path, monkeypatch):
    from ood_repro import eval2_positions
    source, pool, judge = [tmp_path / name for name in ("source", "pool", "judge")]
    for p in (source, pool, judge):
        p.mkdir()
    annotations = judge / "annotations.jsonl"
    annotations.write_text("")
    (judge / "judge-manifest.json").write_text('{"complete": false}')
    monkeypatch.setattr(eval2_positions, "load_study", lambda *args: ({"calibration": []}, [], {}))
    monkeypatch.setattr(sys, "argv", ["positions", "--pool", str(pool), "--test", str(source),
        "--split", "unused", "--annotations", str(annotations), "--output", str(tmp_path / "analysis")])
    with pytest.raises(ValueError, match="incomplete"):
        eval2_positions.main()
