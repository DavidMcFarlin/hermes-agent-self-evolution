"""Tests for artifact delivery into a hermes-agent git checkout.

All tests operate on a throwaway local git repo — nothing is pushed and the
``gh`` CLI is never invoked (open_pr defaults to False).
"""

import subprocess
from pathlib import Path

from evolution.core.delivery import deliver_artifact, is_git_repo


def _run(repo, *args):
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _init_repo(tmp_path: Path, *, with_passing_test=True, failing=False) -> Path:
    repo = tmp_path / "hermes-agent"
    repo.mkdir()
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "t@t.local")
    _run(repo, "config", "user.name", "Tester")
    _run(repo, "config", "commit.gpgsign", "false")
    (repo / "skills" / "demo").mkdir(parents=True)
    (repo / "skills" / "demo" / "SKILL.md").write_text("---\nname: demo\n---\noriginal\n")
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    if failing:
        (tests_dir / "test_x.py").write_text("def test_x():\n    assert False\n")
    elif with_passing_test:
        (tests_dir / "test_x.py").write_text("def test_x():\n    assert True\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "init")
    return repo


def test_non_git_path_is_reported_not_raised(tmp_path):
    result = deliver_artifact(
        hermes_repo=tmp_path / "not-a-repo",
        relative_path="skills/demo/SKILL.md",
        content="x",
        commit_message="msg",
        run_tests=False,
    )
    assert result.delivered is False
    assert "not a git repository" in result.message


def test_unsafe_relative_path_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    result = deliver_artifact(
        hermes_repo=repo,
        relative_path="../escape.md",
        content="x",
        commit_message="msg",
        run_tests=False,
    )
    assert result.delivered is False
    assert "unsafe" in result.message


def test_delivery_creates_branch_and_commit_without_touching_worktree(tmp_path):
    repo = _init_repo(tmp_path)
    head_before = _run(repo, "rev-parse", "HEAD").strip()

    result = deliver_artifact(
        hermes_repo=repo,
        relative_path="skills/demo/SKILL.md",
        content="---\nname: demo\n---\nEVOLVED BODY\n",
        commit_message="evolve: improve demo skill",
        run_tests=False,
    )

    assert result.delivered is True
    assert result.branch and result.commit
    # The user's checked-out HEAD / working tree is untouched.
    assert _run(repo, "rev-parse", "HEAD").strip() == head_before
    assert (repo / "skills" / "demo" / "SKILL.md").read_text() == "---\nname: demo\n---\noriginal\n"
    # The new content lives on the delivery branch.
    blob = _run(repo, "show", f"{result.branch}:skills/demo/SKILL.md")
    assert "EVOLVED BODY" in blob


def test_delivery_runs_test_gate_and_ships_on_pass(tmp_path):
    repo = _init_repo(tmp_path, with_passing_test=True)
    result = deliver_artifact(
        hermes_repo=repo,
        relative_path="skills/demo/SKILL.md",
        content="evolved",
        commit_message="evolve demo",
        run_tests=True,
    )
    assert result.tests_passed is True
    assert result.delivered is True


def test_delivery_aborts_and_discards_branch_when_tests_fail(tmp_path):
    repo = _init_repo(tmp_path, failing=True)
    result = deliver_artifact(
        hermes_repo=repo,
        relative_path="skills/demo/SKILL.md",
        content="evolved",
        commit_message="evolve demo",
        run_tests=True,
    )
    assert result.delivered is False
    assert result.tests_passed is False
    # Branch must not linger after a failed gate.
    branches = _run(repo, "branch", "--list", result.branch)
    assert branches.strip() == ""


def test_no_dangling_worktrees(tmp_path):
    repo = _init_repo(tmp_path)
    deliver_artifact(
        hermes_repo=repo,
        relative_path="skills/demo/SKILL.md",
        content="evolved",
        commit_message="evolve demo",
        run_tests=False,
    )
    # `git worktree list` should only show the main worktree.
    out = _run(repo, "worktree", "list")
    assert len(out.strip().splitlines()) == 1


def test_is_git_repo(tmp_path):
    assert is_git_repo(_init_repo(tmp_path)) is True
    assert is_git_repo(tmp_path / "nope") is False
