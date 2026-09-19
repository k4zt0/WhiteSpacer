import json

from scripts.prepare_security_corpus import Example, safety_examples, write_splits


def test_example_has_provenance_and_chat_shape():
    row = Example("source", "id", "question", "answer").row()
    assert row["source_id"] == "id"
    assert [message["role"] for message in row["messages"]] == [
        "system",
        "user",
        "assistant",
    ]


def test_write_splits_deduplicates(tmp_path):
    example = Example("source", "id", "question", "answer")
    write_splits([example, example], tmp_path, validation_ratio=0, seed=42)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["total"] == 1
    assert manifest["train"] == 1


def test_safety_examples_redirect_harmful_requests():
    examples = list(safety_examples())
    assert len(examples) >= 5
    assert all("authorized" in example.assistant for example in examples)

