"""Deliver an evolved artifact back to the hermes-agent repository.

This closes the loop that the README advertises ("Best variant ──► PR"). The
previous pipeline stopped at ``output/`` and never produced a reviewable change.

Delivery is deliberately layered so the dangerous, outward-facing parts are
opt-in:

1. Always (safe, local, fully testable): create a fresh ``git worktree`` on a
   new branch, write the evolved artifact into place, optionally run the
   hermes-agent test suite as a gate, and commit. The branch is left in the
   repository for a human to inspect. Nothing is pushed.
2. Only when ``open_pr=True`` (outward-facing): push the branch to ``origin``
   and open a pull request via the ``gh`` CLI. Any failure here is reported,
   not fatal — the local branch still exists.

The test-suite gate (guardrail #1) runs against the artifact *as actually
applied* inside the worktree, which is the only place the gate is meaningful.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class DeliveryResult:
    """Outcome of an artifact delivery attempt."""

    delivered: bool
    message: str
    branch: Optional[str] = None
    commit: Optional[str] = None
    pr_url: Optional[str] = None
    tests_passed: Optional[bool] = None
    test_output: str = ""
    steps: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "delivered": self.delivered,
            "message": self.message,
            "branch": self.branch,
            "commit": self.commit,
            "pr_url": self.pr_url,
            "tests_passed": self.tests_passed,
            "steps": list(self.steps),
        }


class DeliveryError(RuntimeError):
    """Raised for unexpected git failures during delivery."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise DeliveryError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc


def is_git_repo(path: Path) -> bool:
    """Return True if ``path`` is inside a git work tree."""
    if not path.exists():
        return False
    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def has_command(name: str) -> bool:
    return shutil.which(name) is not None


def deliver_artifact(
    *,
    hermes_repo: Path,
    relative_path: str,
    content: str,
    commit_message: str,
    branch: Optional[str] = None,
    base_ref: str = "HEAD",
    run_tests: bool = True,
    test_timeout_seconds: int = 600,
    open_pr: bool = False,
    pr_title: Optional[str] = None,
    pr_body: str = "",
) -> DeliveryResult:
    """Apply ``content`` to ``relative_path`` in ``hermes_repo`` on a new branch.

    Returns a :class:`DeliveryResult`. Never raises for the *expected* failure
    modes (not a git repo, tests failed); only genuinely unexpected git errors
    propagate as :class:`DeliveryError`.
    """
    repo = Path(hermes_repo).resolve()
    steps: List[str] = []

    if not is_git_repo(repo):
        return DeliveryResult(
            delivered=False,
            message=f"{repo} is not a git repository; cannot open a PR. "
            "Set HERMES_AGENT_REPO to a git checkout to enable delivery.",
        )

    # Reject paths that escape the repository root.
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        return DeliveryResult(
            delivered=False,
            message=f"refusing unsafe relative_path: {relative_path!r}",
        )

    if branch is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = rel.stem.replace(" ", "-")
        branch = f"evolve/{slug}-{stamp}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="hermes-deliver-"))
    worktree = tmp_dir / "wt"
    branch_created = False
    try:
        _git(repo, "worktree", "add", "-b", branch, str(worktree), base_ref)
        branch_created = True
        steps.append(f"created branch {branch} via worktree")

        target = worktree / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        steps.append(f"wrote {relative_path}")

        tests_passed: Optional[bool] = None
        test_output = ""
        if run_tests:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q"],
                cwd=str(worktree),
                capture_output=True,
                text=True,
                timeout=test_timeout_seconds,
            )
            tests_passed = proc.returncode == 0
            test_output = (proc.stdout or "")[-3000:]
            steps.append(f"ran test suite: {'passed' if tests_passed else 'FAILED'}")
            if not tests_passed:
                # Roll the branch back out — a failing variant must not ship.
                return DeliveryResult(
                    delivered=False,
                    message="evolved artifact failed the hermes-agent test suite; "
                    "branch discarded",
                    branch=branch,
                    tests_passed=False,
                    test_output=test_output,
                    steps=steps,
                )

        _git(worktree, "add", str(rel))
        # Commit with an explicit identity so this works in clean CI checkouts.
        _git(
            worktree,
            "-c",
            "user.name=Hermes Evolution",
            "-c",
            "user.email=evolution@hermes.local",
            "commit",
            "-m",
            commit_message,
        )
        steps.append("committed change")
        sha = _git(worktree, "rev-parse", "HEAD").stdout.strip()

        pr_url: Optional[str] = None
        if open_pr:
            pr_url, pr_steps = _open_pull_request(
                worktree,
                branch=branch,
                title=pr_title or commit_message.splitlines()[0],
                body=pr_body,
            )
            steps.extend(pr_steps)

        return DeliveryResult(
            delivered=True,
            message="artifact delivered to a new branch"
            + (f" and PR opened: {pr_url}" if pr_url else " (no PR pushed)"),
            branch=branch,
            commit=sha,
            pr_url=pr_url,
            tests_passed=tests_passed,
            test_output=test_output,
            steps=steps,
        )
    finally:
        # Always remove the temporary worktree. Keep the branch only if we
        # successfully committed to it; otherwise delete it so we don't litter.
        _git(repo, "worktree", "remove", "--force", str(worktree), check=False)
        committed = any(s == "committed change" for s in steps)
        if branch_created and not committed:
            _git(repo, "branch", "-D", branch, check=False)
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _open_pull_request(
    worktree: Path,
    *,
    branch: str,
    title: str,
    body: str,
) -> tuple[Optional[str], List[str]]:
    """Push the branch and open a PR via gh. Best-effort; never raises."""
    steps: List[str] = []
    if not has_command("gh"):
        steps.append("gh CLI not found; skipped PR (branch left local)")
        return None, steps

    push = _git(worktree, "push", "-u", "origin", branch, check=False)
    if push.returncode != 0:
        steps.append(f"git push failed: {push.stderr.strip()[:200]}")
        return None, steps
    steps.append("pushed branch to origin")

    proc = subprocess.run(
        ["gh", "pr", "create", "--head", branch, "--title", title, "--body", body or title],
        cwd=str(worktree),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        steps.append(f"gh pr create failed: {proc.stderr.strip()[:200]}")
        return None, steps
    url = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
    steps.append(f"opened PR: {url}")
    return url, steps
