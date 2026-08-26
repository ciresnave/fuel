#!/bin/sh
# Install Fuel's version-controlled git hooks.
#
#     sh scripts/install-hooks.sh
#
# Copies scripts/hooks/* into the repository's COMMON hook directory, which is
# what git actually runs.
#
# ---------------------------------------------------------------------------
# WHY A COPY, AND NOT `git config core.hooksPath scripts/hooks`
# ---------------------------------------------------------------------------
# That one-liner is the obvious fix and it is WRONG here. It was tried on
# 2026-08-26 and silently disabled every hook in the repository for about a
# minute.
#
# `core.hooksPath` is a SINGLE value in the common .git/config, shared by every
# worktree. But a RELATIVE path resolves against each worktree's own top level,
# so one setting means twenty-eight different things. At the moment it was
# tried, exactly ONE of Fuel's 28 worktrees contained scripts/hooks/pre-commit
# -- the rest are behind main by 7, 12, 152 commits -- so 27 of 28 would have
# resolved core.hooksPath to a directory that does not exist.
#
# Git accepts a hooksPath pointing at nothing and runs NO HOOKS. No error, no
# warning. The gaps-table gate would have been removed from 27 worktrees and
# nothing would have reported it.
#
# That is the same defect the hook source's own header warns about, one level
# up: it trades "untracked but universally present" for "tracked but present in
# one checkout of twenty-eight". A missing gate is indistinguishable from a
# passing one.
#
# Copying into the common hook dir keeps both properties: the SOURCE is version
# controlled, reviewable and present in a fresh clone; the INSTALLED copy is
# uniform across every worktree because they all share one .git.
set -e

src_dir="$(CDPATH= cd -- "$(dirname -- "$0")/hooks" && pwd)"

# --git-common-dir is the shared .git of the main checkout, correct even when
# this is run from a linked worktree (where --git-dir is .git/worktrees/<name>).
common="$(git rev-parse --git-common-dir)"
case "$common" in
    /*|[A-Za-z]:*) ;;
    *) common="$(CDPATH= cd -- "$common" && pwd)" ;;
esac
dst_dir="$common/hooks"

mkdir -p "$dst_dir"

for src in "$src_dir"/*; do
    [ -f "$src" ] || continue
    name="$(basename "$src")"
    dst="$dst_dir/$name"

    if [ -f "$dst" ] && ! cmp -s "$src" "$dst"; then
        # Never clobber a differing hook silently -- it may be someone's local
        # work, and the whole point of this file is that a hook disappearing
        # without a sound is the failure mode.
        backup="$dst.replaced-$(git rev-parse --short HEAD)"
        cp "$dst" "$backup"
        echo "  backed up existing $name -> $(basename "$backup")"
    fi

    cp "$src" "$dst"
    chmod +x "$dst" 2>/dev/null || true
    echo "  installed $name -> $dst"
done

echo ""
echo "Done. These hooks now run for EVERY worktree sharing this checkout,"
echo "because they all share one common .git directory."
echo ""
echo "Re-run this after pulling a change to scripts/hooks/ -- the installed"
echo "copy does not update itself, and a stale hook is a wrong action rather"
echo "than a wrong answer."
