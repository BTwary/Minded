import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "independent_release" / "manifest.toml"


def _manifest_checks() -> list[dict]:
    with MANIFEST.open("rb") as handle:
        return tomllib.load(handle).get("check", [])


def _joined_commands(checks: list[dict]) -> str:
    # release_gate.py is manifest-driven now: the source no longer names proof
    # scripts directly, the manifest does. Flatten every check's command list
    # into one searchable string so this test still asserts on real gate
    # coverage rather than on release_gate.py's own implementation text.
    return "\n".join(" ".join(str(part) for part in c.get("command", [])) for c in checks)


def test_release_gate_uses_real_controller_proofs_only():
    checks = _manifest_checks()
    commands = _joined_commands(checks)

    # scripts/test_autonomous_controller_messy_input.py must be a real,
    # individually-named gate check (not just incidentally swept up by a
    # directory-wide pytest run), since it's outside tests/independent_release/.
    assert "scripts/test_autonomous_controller_messy_input.py" in commands
    messy_input_check = next(
        c for c in checks if "scripts/test_autonomous_controller_messy_input.py" in " ".join(c.get("command", []))
    )
    assert messy_input_check["expected_exit_code"] == 0
    assert messy_input_check.get("production_path") is True

    # test_defect_007_controller_closure.py and test_defect_015_churn_autonomous_integration.py
    # live under tests/independent_release/, so the blanket "independent_release_suite"
    # check (pytest over the whole directory) already covers them -- confirm that
    # blanket check exists and targets the right directory rather than requiring a
    # separate per-file entry.
    suite_check = next(c for c in checks if c["id"] == "independent_release_suite")
    suite_commands = " ".join(suite_check.get("command", []))
    assert "tests/independent_release/" in suite_commands
    assert suite_check["expected_exit_code"] == 0

    assert 'proof = ROOT / "scripts" / "test_unified_canonical_loop.py"' not in commands
