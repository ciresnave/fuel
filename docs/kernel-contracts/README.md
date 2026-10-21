# Moved

The kernel contracts (`*.fkc.md`) now live in [`fuel-dispatch/kernel-contracts/`](../../fuel-dispatch/kernel-contracts/README.md).

They are embedded with `include_str!`, and a published crate contains only its own directory, so they must sit inside `fuel-dispatch`. Older docs (decisions log, session prompts, plans) still cite the former `docs/kernel-contracts/` path; read it as `fuel-dispatch/kernel-contracts/`.
