#!/usr/bin/env python3
"""Find benchmark directories that have changes relative to a git reference."""

import json
import os
import re
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

# Matches e.g. `    name = "DRUNet"` inside a benchopt Solver class.
SOLVER_NAME_RE = re.compile(r"""^\s*name\s*=\s*(['"])(.*?)\1""", re.MULTILINE)


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
        d
        for d in dirs
        if any(f.startswith(d + "/") or f.startswith(d + os.sep) for f in changed_files)
    ]


def parse_solver_name(path: Path) -> str | None:
    """Extract the `name` class attribute from a benchopt solver file."""
    try:
        text = path.read_text()
    except OSError:
        return None
    match = SOLVER_NAME_RE.search(text)
    return match.group(2) if match else None


def compute_solver_filters(
    dirs: list[str], changed_files: set[str], root: Path
) -> dict[str, list[str]]:
    """Compute, for each benchmark dir, which solvers to restrict a run to.
    """
    filters: dict[str, list[str]] = {}
    for d in dirs:
        solver_prefix = d + "/solvers/"
        dir_changed = {f for f in changed_files if f.startswith(d + "/")}

        if not dir_changed or any(
            not f.startswith(solver_prefix) for f in dir_changed
        ):
            filters[d] = []
            continue

        names = []
        for f in dir_changed:
            if not f.endswith(".py") or Path(f).name == "__init__.py":
                continue
            name = parse_solver_name(root / f)
            if name is None:
                names = []
                break
            names.append(name)

        filters[d] = sorted(set(names))
    return filters


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(description="Find benchmarks in sub-repo")
    parser.add_argument(
        "--all", action="store_true", help="Force to run all benchmarks"
    )
    args = parser.parse_args()

    root = Path.cwd()
    repo = Repo(root)

    # Retrieve all benchmark directories and setup dispatch filters based
    # on dispatch context and git reference range (if applicable)
    all_dirs = find_benchmark_dirs(root)
    dispatch_benchmark_dir = os.environ.get("DISPATCH_BENCHMARK_DIR", "")
    ref_range = get_ref_range(repo)

    if dispatch_benchmark_dir:
        # If a specific benchmark dir is provided via dispatch, directly include it.
        assert dispatch_benchmark_dir in all_dirs, (
            f"Provided BENCHMARK_DIR '{dispatch_benchmark_dir}' is not a valid benchmark.\n"
            "Valid values are:\n- " + "\n- ".join(all_dirs)
        )
        filtered_dirs = [dispatch_benchmark_dir]
        solver_filters = {}
    elif ref_range and not args.all:
        base, head = ref_range
        changed_files = get_changed_files(repo, base, head)
        filtered_dirs = filter_changed_dirs(all_dirs, changed_files)
        solver_filters = compute_solver_filters(filtered_dirs, changed_files, root)
    else:
        # No ref_range (e.g., schedule/tag/create): include all benchmarks
        filtered_dirs = all_dirs
        solver_filters = {}

    # Output as JSON
    print(f"Found benchmark directories:\n{filtered_dirs}")
    print(f"Solver filters (empty list means run all solvers):\n{solver_filters}")
    result = json.dumps(filtered_dirs)

    # If running in GitHub Actions, set the output
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"dirs={result}\nfound_benchmarks={len(filtered_dirs) > 0}\n")
            f.write(f"solver-filters={json.dumps(solver_filters)}\n")


if __name__ == "__main__":
    main()
