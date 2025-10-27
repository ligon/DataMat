#!/usr/bin/env python3
"""Generate GitHub-friendly release notes from git history."""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import subprocess
import sys
import textwrap
from collections.abc import Iterable, Sequence

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_DEFAULT = "https://github.com/ligon/DataMat"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GitError(RuntimeError):
    """Raised when a git command fails."""


def run_git(args: Sequence[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GitError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def resolve_repo_url() -> str:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject_path.read_text())
        urls = data.get("tool", {}).get("poetry", {}).get("urls", {})
        repo_url = urls.get("Repository") or urls.get("Homepage")
        return repo_url or REPO_DEFAULT
    except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError):
        return REPO_DEFAULT


def ref_exists(ref: str) -> bool:
    try:
        run_git(["rev-parse", "--verify", ref])
    except GitError:
        return False
    return True


def resolve_version() -> str:
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject_path.read_text())
    return data["tool"]["poetry"]["version"]


def previous_tag(ref: str) -> str | None:
    try:
        return run_git(["describe", "--tags", "--abbrev=0", f"{ref}^"])
    except GitError:
        return None


def gather_commits(
    start: str | None,
    end: str,
    include_merges: bool,
) -> list[tuple[str, str, str]]:
    if start and not ref_exists(start):
        start = None
    if not ref_exists(end):
        raise GitError(f"Unknown git reference: {end}")
    range_spec = f"{start}..{end}" if start else end
    args = ["log", range_spec, "--pretty=format:%h%x1f%s%x1f%an"]
    if not include_merges:
        args.append("--no-merges")
    output = run_git(args)
    commits: list[tuple[str, str, str]] = []
    if not output:
        return commits
    for line in output.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 3:
            commits.append((parts[0], parts[1], parts[2]))
    return commits


def format_release_notes(
    version: str,
    tag: str,
    compare_url: str | None,
    commits: Iterable[tuple[str, str, str]],
    release_date: dt.date,
    contributors: Iterable[str],
) -> str:
    notes: list[str] = []
    notes.append(f"# DataMat {version}")
    notes.append("")
    notes.append(f"Released {release_date.isoformat()}")
    notes.append("")
    if compare_url:
        notes.append(f"[Compare changes]({compare_url})")
        notes.append("")
    notes.append("## Changes")
    notes.append("")
    seen_any = False
    for short_hash, subject, author in commits:
        seen_any = True
        notes.append(f"- {subject} (`{short_hash}` — {author})")
    if not seen_any:
        notes.append("- No code changes in this interval.")
    notes.append("")
    unique_contributors = sorted(set(contributors))
    if unique_contributors:
        notes.append("## Contributors")
        notes.append("")
        for contributor in unique_contributors:
            notes.append(f"- {contributor}")
        notes.append("")
    notes.append("## Publishing Checklist")
    notes.append("")
    notes.extend(
        textwrap.dedent(
            """
            - [ ] Tag pushed: `{tag}`
            - [ ] PyPI release published
            - [ ] Documentation updated (MkDocs/GitHub Pages)
            """
        )
        .strip()
        .format(tag=tag)
        .splitlines()
    )
    notes.append("")
    return "\n".join(notes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate Markdown release notes from git history."
    )
    parser.add_argument(
        "--from-tag",
        help="Base tag for comparison. Defaults to the previous tag before --to-ref.",
    )
    parser.add_argument(
        "--to-ref",
        default="HEAD",
        help="Target ref (default: HEAD).",
    )
    parser.add_argument(
        "--include-merges",
        action="store_true",
        help="Include merge commits in the changelog.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Write the notes to the specified file instead of stdout.",
    )
    args = parser.parse_args(argv)

    version = resolve_version()
    tag = f"v{version}"
    to_ref = args.to_ref
    from_tag = args.from_tag or previous_tag(to_ref)

    commits = gather_commits(from_tag, to_ref, args.include_merges)
    contributors = [author for _, _, author in commits]

    repo_url = resolve_repo_url()
    compare_url = (
        f"{repo_url}/compare/{from_tag}...{to_ref}"
        if from_tag
        else f"{repo_url}/releases/tag/{tag}"
    )

    notes = format_release_notes(
        version=version,
        tag=tag,
        compare_url=compare_url,
        commits=commits,
        release_date=dt.date.today(),
        contributors=contributors,
    )

    if args.output:
        args.output.write_text(notes + "\n")
    else:
        sys.stdout.write(notes + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
