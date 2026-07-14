# Baracuda ask — flip `SEAM_MAGIC` to `0x4D41_4553` in lockstep (outbound 2026-07-14)

**Status (2026-07-14): RESOLVED — the two seeds are byte-identical again.** Fuel fixed its side (below); baracuda adopted the same value + explicit `reserved1` in lockstep (reply in §5). Only the read-side nonzero-reserved reject remains deferred on baracuda (no live reader yet; lands with their `parse_hello`/negotiate work) — harmless, since baracuda's write-side is now zeroed.**Historical (2026-07-14): FIXED on Fuel's side, pending baracuda lockstep.**
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

## 5. Baracuda reply (inbound 2026-07-14) — DONE in lockstep

Baracuda confirmed both changes landed in `baracuda-seam`, byte-identical to Fuel's seeds:

1. **`SEAM_MAGIC` → `0x4D41_4553`** (was `0x5345_414D`). Value-only; no layout change, no `envelope_version` bump, negotiate/validate untouched. On-wire offset-0 bytes now `53 45 41 4D` = `"SEAM"`, matching Fuel's `1849bc9a` + KISS-ANNOUNCE §6.1-0004.
2. **`reserved1: [u8; 6]`** (offsets 42..48) now an explicit field, zeroed on write, replacing the implicit `#[repr(C)]` padding. Frozen 56-byte layout unchanged (`size_of == 56` compile-time assert holds); mirrors the C reference in `kernel-seam-interop.md` §3.1.

Verification (their side): `seam_magic_wire_bytes_spell_seam` (asserts `== 0x4D41_4553` AND `to_le_bytes() == b"SEAM"`) + `reserved_fields_are_zeroed` (`reserved` + `reserved1` zero in the advertised `SeamHello`); `envelope_is_56_bytes` + the C-ABI out-param test still pass; 5/5 green, clippy clean.

**Deferred half (their side):** the read-side hard-reject of a nonzero `reserved1`/`reserved` (§6.2-0011, Fuel's `SeamError::ReservedNonZero`) has no home yet — `baracuda-seam` has no live handshake *reader* (only the provider-side `baracuda_hello`/`baracuda_seam_hello` out-param fill). It lands with baracuda's `parse_hello` + `negotiate = max(L∩R)` work (their Announce-seed completion item), matching Fuel's `validate_rejects_nonzero_reserved` semantics then. Baracuda's write-side is zeroed now, so nothing it emits trips Fuel's reader. **This closes the lockstep ask; the read-side reject is tracked on baracuda's Announce-seed completion, not here.**
