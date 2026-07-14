//! Kernel-Seam handshake envelope + profile negotiation (Profile v1,
//! `docs/specs/kernel-seam-interop.md` §3).
//!
//! A connection **negotiates a profile before any tensor or kernel crosses the
//! seam** — the piece that lets Fuel, Baracuda, Vulkane, and future ecosystems
//! support multiple profile versions over time without a lockstep flag-day.
//! [`SeamHello`] is the frozen C-ABI envelope by which each side advertises its
//! supported profiles; [`negotiate`] runs §3.2 ("highest mutually-supported
//! wins, hard-fail on disjoint"). The struct layout is **frozen forever** and
//! cross-checked by the size/offset asserts in the tests below — the same
//! discipline FDX uses for its `#[repr(C)]` structs.
//!
//! These are the frozen wire types + their pure negotiation helpers, kept in
//! their own dependency-free crate — separate from the region grammar
//! (`fuel-kernel-seam-types`'s `OpTag`/`OpAttrs`/`PatternNode`) so a provider
//! that only speaks capability negotiation (KISS-Announce) never needs to pull
//! in the grammar (KISS-Grammar) to depend on this. The FFI call that obtains a
//! provider's envelope (`int baracuda_seam_hello(SeamHello* out)`) and the live
//! handshake driver live in the protocol crate / the backend glue that links
//! the provider; this crate is pure + portable.

/// `"SEAM"` — the envelope magic; never changes (§3.1).
///
/// Chosen so the little-endian on-wire bytes at offset 0 read `53 45 41 4D` =
/// ASCII `S E A M`. That requires the *numeric* constant `0x4D41_4553` (the
/// bytes reversed): a u32 whose big-endian spelling is "SEAM" (`0x5345_414D`)
/// serializes little-endian to `4D 41 45 53` = "MAES" — the endianness
/// inversion this pins down. Matches KISS-ANNOUNCE §6.1-0004
/// (`magic == 0x4D414553` read as an LE u32).
pub const SEAM_MAGIC: u32 = 0x4D41_4553;
/// The envelope's own version; designed never to bump (§3.1).
pub const SEAM_ENVELOPE_VERSION: u8 = 1;
/// Fixed cap on simultaneously-advertised profiles (§3.1).
pub const SEAM_MAX_PROFILES: usize = 16;

/// Profile v1 — the bundled FDX + FKC version (§2). The only profile that
/// exists today; Fuel advertises `[PROFILE_V1]`.
pub const PROFILE_V1: u16 = 1;

// ---- Capability bits (§3.4) ------------------------------------------------
// Profile v1 adopts the FDX `BackendProbe` tokens as the low bits and reserves
// higher bits for FKC-/JIT-level optional features. The negotiated set is
// `local & remote`; a feature neither side flags is simply not used.

/// FDX base sidecar support (`DlpackExtV1`).
pub const SEAM_CAP_FDX_V1: u64 = 1 << 0;
/// FDX MX block-scaled quant (`DlpackExtMx`).
pub const SEAM_CAP_FDX_MX: u64 = 1 << 1;
/// FDX GGML inline-block quant (`DlpackExtGgml`).
pub const SEAM_CAP_FDX_GGML: u64 = 1 << 2;
/// FDX affine (int/float) quant (`DlpackExtAffine`).
pub const SEAM_CAP_FDX_AFFINE: u64 = 1 << 3;
/// FDX symbolic extents (`DlpackExtSymbolic`).
pub const SEAM_CAP_FDX_SYMBOLIC: u64 = 1 << 4;
/// FDX indexed-residency / gather (`DlpackExtGather`).
pub const SEAM_CAP_FDX_GATHER: u64 = 1 << 5;
/// JIT-on-request endpoint (§5); a party may support Profile v1's FDX+FKC
/// without this.
pub const SEAM_CAP_JIT_ON_REQUEST: u64 = 1 << 16;

/// The negotiation envelope — a FIXED-SIZE C-ABI POD, frozen for all time
/// (§3.1). 56 bytes; field offsets are frozen and asserted (see the tests).
/// Providers fill a caller-allocated `SeamHello` via the out-param entry point
/// `int baracuda_seam_hello(SeamHello* out)`; Fuel reads it and runs
/// [`negotiate`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeamHello {
    /// == [`SEAM_MAGIC`].
    pub magic: u32,
    /// == [`SEAM_ENVELOPE_VERSION`].
    pub envelope_version: u8,
    /// == 0.
    pub reserved: [u8; 3],
    /// Number of valid entries in `profiles` (≤ [`SEAM_MAX_PROFILES`]).
    pub profiles_len: u16,
    /// Ascending; entries `[profiles_len..]` are 0.
    pub profiles: [u16; SEAM_MAX_PROFILES],
    /// Alignment padding between `profiles` (ends at offset 42) and the
    /// 8-byte-aligned `capabilities` (offset 48) made **explicit** so it is a
    /// managed field — zeroed by [`advertise`] and hard-rejected when nonzero
    /// by [`SeamHello::validate`] — rather than implicit `#[repr(C)]` padding
    /// Rust neither guarantees zeroed nor lets a reader inspect. == 0.
    pub reserved1: [u8; 6],
    /// Optional-feature bitset within the selected profile (§3.4).
    pub capabilities: u64,
}

/// A typed handshake failure — never a panic, never silent coercion (§3.2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SeamError {
    /// `magic != SEAM_MAGIC` — not a seam envelope at all.
    BadMagic(u32),
    /// `envelope_version` is one this build cannot parse.
    UnknownEnvelope(u8),
    /// `profiles_len > SEAM_MAX_PROFILES`.
    TooManyProfiles(u16),
    /// The advertised profile list is not strictly ascending (§3.1).
    NotAscending,
    /// A reserved field (`reserved` or `reserved1`) carried a nonzero byte.
    /// Reserved bytes are pinned to zero so a foreign or garbage-padded
    /// envelope is hard-rejected rather than silently accepted (§3.1).
    ReservedNonZero,
    /// No mutually-supported profile — the connection does NOT proceed (§3.2).
    VersionMismatch { local: Vec<u16>, remote: Vec<u16> },
}

impl core::fmt::Display for SeamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SeamError::BadMagic(m) => write!(
                f,
                "seam: bad magic 0x{m:08X} (expected 0x{SEAM_MAGIC:08X}) — not a SeamHello envelope"
            ),
            SeamError::UnknownEnvelope(v) => write!(
                f,
                "seam: envelope_version {v} unknown (this build speaks {SEAM_ENVELOPE_VERSION})"
            ),
            SeamError::TooManyProfiles(n) => write!(
                f,
                "seam: profiles_len {n} exceeds SEAM_MAX_PROFILES {SEAM_MAX_PROFILES}"
            ),
            SeamError::NotAscending => {
                write!(f, "seam: advertised profiles must be strictly ascending")
            }
            SeamError::ReservedNonZero => {
                write!(f, "seam: a reserved field carried a nonzero byte (must be zero)")
            }
            SeamError::VersionMismatch { local, remote } => write!(
                f,
                "seam: no mutually-supported profile (local {local:?}, remote {remote:?}); \
                 connection does not proceed"
            ),
        }
    }
}

impl std::error::Error for SeamError {}

/// The §3.2 negotiation result for a connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Negotiated {
    /// The selected profile — the highest mutually-supported one.
    pub profile: u16,
    /// `local.capabilities & remote.capabilities` (§3.4).
    pub capabilities: u64,
}

impl SeamHello {
    /// Build Fuel's own advertisement from an ascending profile list + capability
    /// bits. Never panics: a list longer than the cap is truncated (a Fuel-side
    /// caller bug surfaced by the `debug_assert`s rather than a release abort).
    pub fn advertise(profiles: &[u16], capabilities: u64) -> Self {
        debug_assert!(
            profiles.len() <= SEAM_MAX_PROFILES,
            "advertise: more than {SEAM_MAX_PROFILES} profiles",
        );
        debug_assert!(
            profiles.windows(2).all(|w| w[0] < w[1]),
            "advertise: profiles must be strictly ascending",
        );
        let n = profiles.len().min(SEAM_MAX_PROFILES);
        let mut arr = [0u16; SEAM_MAX_PROFILES];
        arr[..n].copy_from_slice(&profiles[..n]);
        Self {
            magic: SEAM_MAGIC,
            envelope_version: SEAM_ENVELOPE_VERSION,
            reserved: [0; 3],
            profiles_len: n as u16,
            profiles: arr,
            reserved1: [0; 6],
            capabilities,
        }
    }

    /// Fuel's standard advertisement: Profile v1 with the given capability bits.
    pub fn fuel(capabilities: u64) -> Self {
        Self::advertise(&[PROFILE_V1], capabilities)
    }

    /// Validate the frozen envelope and return the advertised profiles slice. A
    /// remote `SeamHello` MUST pass this before [`negotiate`] consumes it.
    pub fn validate(&self) -> Result<&[u16], SeamError> {
        if self.magic != SEAM_MAGIC {
            return Err(SeamError::BadMagic(self.magic));
        }
        if self.envelope_version != SEAM_ENVELOPE_VERSION {
            return Err(SeamError::UnknownEnvelope(self.envelope_version));
        }
        // Fixed-prefix reserved bytes are pinned to zero: a foreign or garbage-
        // padded envelope is hard-rejected, not silently accepted.
        if self.reserved != [0; 3] || self.reserved1 != [0; 6] {
            return Err(SeamError::ReservedNonZero);
        }
        let n = self.profiles_len as usize;
        if n > SEAM_MAX_PROFILES {
            return Err(SeamError::TooManyProfiles(self.profiles_len));
        }
        let live = &self.profiles[..n];
        if live.windows(2).any(|w| w[0] >= w[1]) {
            return Err(SeamError::NotAscending);
        }
        Ok(live)
    }
}

/// §3.2: select the **highest mutually-supported** profile (TLS-style), with
/// `capabilities = local & remote`. Hard-fails with [`SeamError::VersionMismatch`]
/// on disjoint advertised sets — the connection does NOT proceed on a guessed
/// version. Both envelopes are validated first.
pub fn negotiate(local: &SeamHello, remote: &SeamHello) -> Result<Negotiated, SeamError> {
    let lp = local.validate()?;
    let rp = remote.validate()?;
    // Both lists ascending; iterate local high→low and take the first that the
    // remote also advertises = max of the intersection.
    match lp.iter().rev().find(|p| rp.contains(p)).copied() {
        Some(profile) => Ok(Negotiated {
            profile,
            capabilities: local.capabilities & remote.capabilities,
        }),
        None => Err(SeamError::VersionMismatch {
            local: lp.to_vec(),
            remote: rp.to_vec(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn seamhello_layout_is_frozen() {
        // The envelope must outlive every profile it negotiates, so its size +
        // field offsets are frozen forever and cross-checked on both sides.
        assert_eq!(size_of::<SeamHello>(), 56, "SeamHello is frozen at 56 bytes");
        assert_eq!(align_of::<SeamHello>(), 8, "u64 capabilities forces 8-byte align");
        assert_eq!(offset_of!(SeamHello, magic), 0);
        assert_eq!(offset_of!(SeamHello, envelope_version), 4);
        assert_eq!(offset_of!(SeamHello, reserved), 5);
        assert_eq!(offset_of!(SeamHello, profiles_len), 8);
        assert_eq!(offset_of!(SeamHello, profiles), 10);
        assert_eq!(offset_of!(SeamHello, reserved1), 42);
        assert_eq!(offset_of!(SeamHello, capabilities), 48);
    }

    #[test]
    fn seam_magic_wire_bytes_spell_seam() {
        // The envelope magic must appear on the wire (little-endian, offset 0)
        // as the ASCII bytes `S E A M` = `53 45 41 4D`, matching KISS-ANNOUNCE
        // §6.1-0004 (`magic == 0x4D414553` when read as an LE u32).
        assert_eq!(
            SEAM_MAGIC.to_le_bytes(),
            *b"SEAM",
            "SEAM_MAGIC must serialize little-endian to the bytes 'SEAM', not 'MAES'"
        );
    }

    #[test]
    fn validate_rejects_nonzero_reserved() {
        // A KISS-conform reader hard-rejects an envelope with nonzero reserved
        // bytes; Fuel's own reader must do the same so a foreign or garbage-
        // padded envelope never slips through.
        let mut bad = SeamHello::fuel(0);
        bad.reserved = [1, 0, 0];
        assert!(
            matches!(bad.validate(), Err(SeamError::ReservedNonZero)),
            "validate() must reject nonzero reserved0 bytes"
        );

        let mut bad1 = SeamHello::fuel(0);
        bad1.reserved1 = [0, 0, 7, 0, 0, 0];
        assert!(
            matches!(bad1.validate(), Err(SeamError::ReservedNonZero)),
            "validate() must reject nonzero reserved1 (alignment-padding) bytes"
        );
    }

    #[test]
    fn advertise_zeroes_all_reserved_padding() {
        // The 6 bytes between `profiles` (ends at 42) and `capabilities` (48)
        // are real struct storage; advertise() MUST zero them so a struct-as-
        // wire envelope never carries uninitialized garbage a foreign reader
        // rejects. Inspect the raw bytes to prove it.
        let hello = SeamHello::fuel(SEAM_CAP_FDX_V1);
        let raw: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &hello as *const SeamHello as *const u8,
                size_of::<SeamHello>(),
            )
        };
        assert!(
            raw[42..48].iter().all(|&b| b == 0),
            "reserved1 alignment padding (bytes 42..48) must be all-zero, got {:?}",
            &raw[42..48]
        );
    }

    #[test]
    fn v1_both_sides_select_profile_1() {
        let local = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_FDX_GGML);
        let remote = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_FDX_MX);
        let n = negotiate(&local, &remote).expect("v1 ∩ v1 negotiates");
        assert_eq!(n.profile, PROFILE_V1);
        // capabilities = local & remote — only the common FDX_V1 bit survives.
        assert_eq!(n.capabilities, SEAM_CAP_FDX_V1);
    }

    #[test]
    fn highest_mutually_supported_wins() {
        let local = SeamHello::advertise(&[1, 2, 3], 0);
        let remote = SeamHello::advertise(&[2, 3, 5], 0);
        assert_eq!(negotiate(&local, &remote).unwrap().profile, 3);
    }

    #[test]
    fn disjoint_profiles_hard_fail() {
        let local = SeamHello::advertise(&[1], 0);
        let remote = SeamHello::advertise(&[2], 0);
        match negotiate(&local, &remote) {
            Err(SeamError::VersionMismatch { local, remote }) => {
                assert_eq!(local, vec![1]);
                assert_eq!(remote, vec![2]);
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn jit_capability_negotiates_to_off_when_one_side_lacks_it() {
        // Fuel supports JIT-on-request; a provider that doesn't → off.
        let fuel = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_JIT_ON_REQUEST);
        let provider = SeamHello::fuel(SEAM_CAP_FDX_V1);
        let n = negotiate(&fuel, &provider).unwrap();
        assert_eq!(n.capabilities & SEAM_CAP_JIT_ON_REQUEST, 0, "JIT not on this connection");
        assert_eq!(n.capabilities & SEAM_CAP_FDX_V1, SEAM_CAP_FDX_V1);
    }

    #[test]
    fn validate_rejects_bad_magic_and_unsorted() {
        let mut bad = SeamHello::fuel(0);
        bad.magic = 0xDEAD_BEEF;
        assert!(matches!(bad.validate(), Err(SeamError::BadMagic(0xDEAD_BEEF))));

        let mut unsorted = SeamHello::advertise(&[1, 2], 0);
        unsorted.profiles[0] = 2;
        unsorted.profiles[1] = 1; // now descending
        assert!(matches!(unsorted.validate(), Err(SeamError::NotAscending)));

        let mut toomany = SeamHello::fuel(0);
        toomany.profiles_len = (SEAM_MAX_PROFILES + 1) as u16;
        assert!(matches!(toomany.validate(), Err(SeamError::TooManyProfiles(_))));
    }
}
