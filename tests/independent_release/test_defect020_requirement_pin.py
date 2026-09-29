from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_sqlglot_is_pinned_in_runtime_requirements():
    assert "sqlglot==30.18.0" in _text("requirements.txt")
    assert "sqlglot==30.18.0" in _text("requirements-server.txt")
    assert "sqlglot==30.18.0" in _text("requirements-hf-space.txt")
    lock = _text("requirements-lock.txt")
    assert "sqlglot==30.18.0" in lock
    assert "ee0f9a9f3e2193e763c326e52dfb377b96fdb4292f6305c3ff2d9a220e71c601" in lock
    assert "e57e1b205e341979d1df5b1212c1435c598a0437e4619e3f428b15d5bc3a5cc6" in lock
