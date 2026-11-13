#!/usr/bin/env python3
"""Publish changed research source from a PACE checkout, excluding run artifacts.

Default: print a dry-run manifest. --push: validate, make a new branch, commit,
and push only source/configuration/documentation changes to RushMaples/graphnet.
Python 3.8+; standard library only. Never imports a model or runs training.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import subprocess
import sys

FORK = "https://github.com/RushMaples/graphnet.git"
REMOTE = "pace-code-fork"
MAX_BYTES = 2 * 1024 * 1024
SOURCE_SUFFIXES = {".py", ".sh", ".sbatch", ".md", ".rst", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".js", ".mjs"}
SOURCE_PREFIXES = ("comparison/", "experiments/", "src/graphnet/")
TOP_FILES = {".gitignore", "pyproject.toml", "setup.py", "setup.cfg", "environment.yml", "environment.yaml", "requirements.txt"}
EXCLUDED_DIRS = {"comparison_runs", "output", "outputs", "results", "logs", "checkpoints", "predictions", "__pycache__", ".git", ".venv", "venv", "node_modules", "backup", "backups"}


def source_path(value):
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        return False
    if any(x.lower() in EXCLUDED_DIRS for x in p.parts[:-1]):
        return False
    if any(x in p.name.lower() for x in (".before", ".bak", "credentials", "secrets")):
        return False
    if value in TOP_FILES:
        return True
    if not value.startswith(SOURCE_PREFIXES):
        return False
    if p.suffix.lower() == ".json" and p.name != "pilot_contract.json" and not any(x in {"config", "configs"} for x in p.parts[:-1]):
        return False
    return p.suffix.lower() in SOURCE_SUFFIXES or (p.name.startswith("requirements") and p.suffix == ".txt")


def git(repo, *args, check=True):
    result = subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode:
        # Do not print configured remotes, which can contain embedded credentials.
        if args[:2] == ("remote", "get-url"):
            raise RuntimeError("Could not read the publishing remote.")
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError("Git command failed: " + " ".join(args[:2]) + "\n" + error)
    return result


def names(repo, *args):
    return {x.decode("utf-8", errors="surrogateescape") for x in git(repo, *args).stdout.split(b"\0") if x}


def manifest(repo):
    staged = names(repo, "diff", "--cached", "--name-only", "-z")
    unsafe_staged = sorted(x for x in staged if not source_path(x))
    if unsafe_staged:
        raise RuntimeError("Non-source files are already staged. They were left untouched:\n" + "\n".join(unsafe_staged) + "\nReview git diff --cached --stat before continuing.")
    changed = staged | names(repo, "diff", "--name-only", "-z") | names(repo, "ls-files", "--others", "--exclude-standard", "-z")
    selected = sorted(x for x in changed if source_path(x))
    excluded = sorted(changed - set(selected))
    for value in selected:
        p = repo / value
        if p.is_symlink():
            raise RuntimeError("Refusing to publish a source symlink: " + value)
        if p.is_file() and p.stat().st_size > MAX_BYTES:
            raise RuntimeError("Source candidate exceeds 2 MiB; review it separately: " + value)
        if p.is_file() and p.suffix == ".py":
            ast.parse(p.read_text(encoding="utf-8-sig"), filename=value)
    return selected, excluded


def ensure_fork(repo):
    old = git(repo, "remote", "get-url", REMOTE, check=False)
    if old.returncode == 0:
        url = old.stdout.decode().strip().rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        if url.lower() not in {"https://github.com/rushmaples/graphnet", "git@github.com:rushmaples/graphnet", "ssh://git@github.com/rushmaples/graphnet"}:
            raise RuntimeError("Remote " + REMOTE + " targets a different repository. No push was attempted.")
    else:
        # Reuse SSH authentication when this checkout already uses it for the fork.
        target = FORK
        for name in git(repo, "remote").stdout.decode().splitlines():
            value = git(repo, "remote", "get-url", name).stdout.decode().strip().rstrip("/")
            canonical = value[:-4] if value.endswith(".git") else value
            if canonical.lower() in {"git@github.com:rushmaples/graphnet", "ssh://git@github.com/rushmaples/graphnet"}:
                target = "git@github.com:RushMaples/graphnet.git"
                break
        git(repo, "remote", "add", REMOTE, target)
    git(repo, "fetch", "--no-tags", REMOTE, "main:refs/remotes/" + REMOTE + "/main")
    pending = git(repo, "rev-list", "HEAD", "--not", "refs/remotes/" + REMOTE + "/main").stdout.decode().splitlines()
    if pending:
        raise RuntimeError("HEAD contains commits not already on the fork's main branch. Review those commits first so an older output commit is not accidentally published. No source files were staged by this helper.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="PACE graphnet checkout; default is the current directory")
    parser.add_argument("--push", action="store_true", help="Commit and push the validated source manifest on a new branch")
    parser.add_argument("--branch", help="Optional new branch name; an existing local branch is never reset")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    top = Path(git(repo, "rev-parse", "--show-toplevel").stdout.decode().strip()).resolve()
    if repo != top:
        raise RuntimeError("--repo must name the root of the graphnet checkout.")
    selected, excluded = manifest(repo)
    print("SOURCE MANIFEST ({} files):".format(len(selected)))
    for value in selected:
        print("  " + value)
    print("Excluded {} non-source changes; predictions, weights, archives and run directories are not staged.".format(len(excluded)))
    if not args.push:
        print("Dry run only. Add --push to publish this source manifest from PACE.")
        return 0
    if not selected:
        print("No new source changes. If a previous push failed after committing, retry: git push -u " + REMOTE + " HEAD")
        return 0
    ensure_fork(repo)
    branch = args.branch or ("research/icemix-code-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    if git(repo, "show-ref", "--verify", "--quiet", "refs/heads/" + branch, check=False).returncode == 0:
        raise RuntimeError("Local branch already exists; choose a new --branch name. It was not reset.")
    git(repo, "checkout", "-b", branch)
    for start in range(0, len(selected), 50):
        git(repo, "add", "-A", "--", *selected[start:start + 50])
    actual = names(repo, "diff", "--cached", "--name-only", "-z")
    if not actual or not actual.issubset(set(selected)) or not all(source_path(x) for x in actual):
        raise RuntimeError("The staged manifest changed unexpectedly. No commit or push was made.")
    git(repo, "diff", "--cached", "--check")
    print(git(repo, "diff", "--cached", "--stat").stdout.decode())
    git(repo, "commit", "-m", "Add directional-density comparisons and PACE research workflows")
    try:
        git(repo, "push", "-u", REMOTE, "HEAD:refs/heads/" + branch)
    except RuntimeError:
        print("The source commit is saved locally on " + branch + ".")
        print("After fixing GitHub authentication/network access, retry: git push -u " + REMOTE + " HEAD")
        raise
    sha = git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    print("Published source commit: " + sha)
    print("Branch: https://github.com/RushMaples/graphnet/tree/" + branch)
    print("Generated outputs and checkpoint binaries remain in their existing PACE locations.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, OSError, SyntaxError) as exc:
        print("STOP: " + str(exc), file=sys.stderr)
        sys.exit(1)
