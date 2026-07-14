# Baracuda ask — flip `SEAM_MAGIC` to `0x4D41_4553` in lockstep (outbound 2026-07-14)

**Fuel-side status (2026-07-14): FIXED on Fuel's side, pending baracuda lockstep.**
Fuel flipped its `SEAM_MAGIC` constant (commit `1849bc9a`, `fuel-kernel-seam-announce`) and
the mirrored C-header `#define` in [`docs/specs/kernel-seam-interop.md`](../specs/kernel-seam-interop.md).
`baracuda-seam` must adopt the **same** value so the two co-developed seeds stay byte-identical.
**Low urgency but do it soon:** no live `baracuda_seam_hello` handshake exists yet, so the two
seeds being briefly out of sync breaks nothing today — but the moment a handshake goes live (or
any KISS-conform / big-endian third party reads a Fuel or baracuda envelope), a mismatched magic
is a hard-reject on first contact. Cheaper to fix now, while both sides are draft and unwired.

---

## 1. The bug — an endianness inversion

Both Evans-Labs seam seeds shipped `SEAM_MAGIC = 0x5345_414D`. That constant spells `"SEAM"` only
when read **big-endian**. But `SeamHello` is a fixed-layout `#[repr(C)]` / POD struct transmitted in
the platform's native (little-endian) byte order, so the `magic` field's on-wire bytes at offset 0
are the u32's least-significant byte first:

```
0x5345_414D  --LE-->  wire bytes  4D 41 45 53  =  "MAES"   (wrong)
0x4D41_4553  --LE-->  wire bytes  53 45 41 4D  =  "SEAM"   (correct)
```

KISS-ANNOUNCE **§6.1-0004** pins it precisely: *"The `magic` field MUST equal `0x4D414553` when read
as a little-endian u32 (on-wire bytes `53 45 41 4D`, ASCII `SEAM`)."* So the numeric constant that
puts `"SEAM"` on the wire is `0x4D41_4553` — the bytes of the intended string reversed.

## 2. The ask

Change `baracuda-seam`'s magic constant to match Fuel's:

```diff
- #define SEAM_MAGIC 0x5345414Du   /* "SEAM" — WRONG: LE wire bytes are "MAES" */
+ #define SEAM_MAGIC 0x4D414553u   /* on-wire bytes 53 45 41 4D = "SEAM" (KISS-ANNOUNCE §6.1-0004) */
```

This is a **value-only** change: no struct layout change, no `envelope_version` bump, and the
validate/negotiate logic is untouched (both sides just compare `magic == SEAM_MAGIC`). Because
neither side has a live handshake, there is no flag-day — but the two constants MUST land as a pair
so the seeds never disagree in a persisted or transmitted envelope.

## 3. While you're there — the `reserved1` alignment field (if you mirror `SeamHello`)

Fuel also made the 6 bytes of alignment padding between `profiles` (ends at offset 42) and the
8-byte-aligned `capabilities` (offset 48) an **explicit** field rather than implicit `#[repr(C)]`
padding, so it can be zeroed and validated (a KISS-conform reader hard-rejects a nonzero reserved
field, §6.2-0011). The frozen 56-byte layout is unchanged. If baracuda constructs or emits a
`SeamHello`:

```c
  uint16_t profiles[SEAM_MAX_PROFILES];  /* offsets 10..42 */
  uint8_t  reserved1[6];                 /* offset 42..48 — MUST be zeroed */
  uint64_t capabilities;                 /* offset 48 */
```

Zero `reserved1` on write and reject a nonzero `reserved1` (and the existing `reserved[3]`) on read.
This is optional/advisory relative to the magic flip (a struct memset already covers it in practice),
but it keeps the two implementations bit-for-bit aligned with the [C reference in
`kernel-seam-interop.md`](../specs/kernel-seam-interop.md).

## 4. References

- KISS-ANNOUNCE §6.1-0004 (magic value) + §6.2-0011 (reserved hard-reject) — github.com/ThinkersJournal/KISS.
- Fuel fix: commit `1849bc9a`, `fuel-kernel-seam-announce/src/lib.rs` (const + `reserved1` field + `SeamError::ReservedNonZero`), tests `seam_magic_wire_bytes_spell_seam` / `validate_rejects_nonzero_reserved`.
- C reference struct + `#define`: [`docs/specs/kernel-seam-interop.md`](../specs/kernel-seam-interop.md) §3.1.
- Context: [`docs/outreach/kiss-conformance-and-divergences.md`](kiss-conformance-and-divergences.md) §1.
