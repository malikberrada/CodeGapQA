from pathlib import Path
from codegap_qa.freeze import freeze, verify_freeze


def test_freeze(tmp_path: Path):
    (tmp_path / "x.txt").write_text("hello", encoding="utf-8")
    freeze(tmp_path)
    assert verify_freeze(tmp_path / "freeze_manifest.json")["ok"]
