# Shipped session prompts (archive)

Session prompts whose work has landed. Kept for historical record — useful
when revisiting the *why* of an architectural decision long after the
implementation memory has faded. The shipped state is reflected in:

- `ROADMAP.md` (per-phase work-item checkboxes)
- Memory files in `~/.claude/projects/.../memory/` (per-shipped-work
  `project_*_shipped.md` entries)
- Commit log

Each shipped prompt's date and corresponding memory entry can be found
by grepping the prompt's filename across `ROADMAP.md` and the memory
directory's index `MEMORY.md`.

## When to move a session prompt out of this archive

If a follow-up session needs to re-open work that was previously declared
shipped (e.g., a bug surfaces, scope expands), move the prompt back to
`../` and add a "Re-opened YYYY-MM-DD: <reason>" header. Don't edit the
original prompt body — its historical state is part of why it's preserved.
