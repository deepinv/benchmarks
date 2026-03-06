#!/usr/bin/env python3
"""Find benchmark directories that have changes relative to a git reference."""

import json
import os
from pathlib import Path

from git import Repo
from git.exc import GitCommandError


def find_benchmark_dirs(root: Path, max_depth: int = 4) -> list[str]:
    """Find all directories containing an objective.py file."""
    dirs = []
    for path in root.rglob("objective.py"):
        # Check depth relative to root
        rel_path = path.relative_to(root)
        if len(rel_path.parts) <= max_depth:
            dirs.append(str(path.parent.relative_to(root)))
    return sorted(dirs)


def get_ref_range(repo: Repo) -> tuple[str, str] | None:
    """Compute the git reference range based on GitHub event type.

    Returns a tuple of (base_commit, head_commit) or None.
    """
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")

    if event_name == "pull_request":
        base_ref = os.environ.get("GITHUB_BASE_REF", "")
        if base_ref:
            # Fetch base branch
            try:
                repo.remotes.origin.fetch(
                    refspec=f"+refs/heads/{base_ref}:"
                            f"refs/remotes/origin/{base_ref}",
                    depth=1,
                    no_tags=True,
                    prune=True,
                )
            except GitCommandError:
                pass
            return (f"origin/{base_ref}", "HEAD")

    elif event_name == "push":
        before = os.environ.get("GITHUB_EVENT_BEFORE", "")
        sha = os.environ.get("GITHUB_SHA", "")
        if before and sha:
            return (before, sha)

    return None


def get_changed_files(repo: Repo, base: str, head: str) -> set[str]:
    """Get all files changed between two commits."""
    try:
        # Get the diff between base and head
        base_commit = repo.commit(base)
        head_commit = repo.commit(head)
        diff = base_commit.diff(head_commit)

        changed = set()
        for diff_item in diff:
            if diff_item.a_path:
                changed.add(diff_item.a_path)
            if diff_item.b_path:
                changed.add(diff_item.b_path)
        print(changed)
        return changed
    except GitCommandError:
        return set()


def filter_changed_dirs(dirs: list[str], changed_files: set[str]) -> list[str]:
    """Filter directories to only include those with changes."""
    return [
        d for d in dirs
        if any(
                f.startswith(d + "/") or f.startswith(d + os.sep)
                for f in changed_files
        )
    ]


def main() -> None:

    import argparse
    parser = argparse.ArgumentParser(description='Find benchmarks in sub-repo')
    parser.add_argument('--all', action="store_true",
                        help='Force to run all benchmarks')
    args = parser.parse_args()

    root = Path.cwd()
    repo = Repo(root)

    # Find all benchmark directories
    all_dirs = find_benchmark_dirs(root)

    # Get reference range for filtering
    ref_range = get_ref_range(repo)

    if ref_range and not args.all:
        base, head = ref_range
        changed_files = get_changed_files(repo, base, head)
        filtered_dirs = filter_changed_dirs(all_dirs, changed_files)
    else:
        # No ref_range (e.g., schedule/tag/create): include all benchmarks
        filtered_dirs = all_dirs

    # Output as JSON
    result = json.dumps(filtered_dirs)
    print(result)

    # If running in GitHub Actions, set the output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"dirs={result}\n")


if __name__ == "__main__":
    main()
