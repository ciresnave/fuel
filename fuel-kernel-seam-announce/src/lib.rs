// SPDX-License-Identifier: MIT OR Apache-2.0
//! Kernel-Seam handshake envelope + profile negotiation (Profile v1,
//! `docs/specs/kernel-seam-interop.md` §3).
//!
//! A connection **negotiates a profile before any tensor or kernel crosses the
//! seam** — the piece that lets Fuel, Baracuda, Vulkane, and future ecosystems
//! support multiple profile versions over time without a lockstep flag-day.
//! [`SeamHello`] is the frozen C-ABI envelope by which each side advertises its
//! supported profiles; [`negotiate_with_role`] runs §3.2 ("highest mutually-supported
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
// `local & remote` for the EXT and SUB axes; the FEAT axis is the provider's word alone
// (KISS-ANNOUNCE-7.3-0003, see [`negotiate_with_role`]).

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
// ---- KISS-ANNOUNCE §7.2 axes and the registry bits Fuel acts on (kiss `spec/announce.md` @4656dc2) ----

/// EXT axis, bits `[0, 24)`: external interchange tokens (what each side *speaks*).
pub const SEAM_CAP_EXT_MASK: u64 = (1 << 24) - 1;
/// FEAT axis, bits `[24, 48)`: provider-level optional features (what the provider *offers*).
pub const SEAM_CAP_FEAT_MASK: u64 = ((1 << 24) - 1) << 24;
/// SUB axis, bits `[48, 64)`: which sub-standards the provider speaks (presence only).
pub const SEAM_CAP_SUB_MASK: u64 = ((1 << 16) - 1) << 48;
/// The bits only the PROVIDER decides: the whole FEAT axis (KISS-ANNOUNCE-7.3-0003 masks by axis,
/// not by recognition, so two readers that recognise different FEAT bits compute the same word).
pub const SEAM_CAP_PROVIDER_ONLY_MASK: u64 = SEAM_CAP_FEAT_MASK;
/// FEAT bit 32 `PROVISION_ON_REQUEST`: the provider builds a kernel on a miss (the JIT endpoint, §5).
pub const SEAM_CAP_PROVISION_ON_REQUEST: u64 = 1 << 32;
/// FEAT bit 33 `CONTRACT_QUERY`: the provider answers contract queries.
pub const SEAM_CAP_CONTRACT_QUERY: u64 = 1 << 33;

/// Which side of the exchange the LOCAL envelope is. Known from the exchange itself, never inferred
/// from a capability bit (the envelope is frozen at 56 bytes and carries no role field).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The local side asks for kernels (Fuel, in the JIT-on-request exchange).
    Consumer,
    /// The local side supplies kernels (a backend such as baracuda).
    Provider,
}

/// The negotiation envelope — a FIXED-SIZE C-ABI POD, frozen for all time
/// (§3.1). 56 bytes; field offsets are frozen and asserted (see the tests).
/// Providers fill a caller-allocated `SeamHello` via the out-param entry point
/// `int baracuda_seam_hello(SeamHello* out)`; Fuel reads it and runs
/// [`negotiate_with_role`].
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
    /// managed field — zeroed by [`crate::SeamHello::advertise`] and hard-rejected when nonzero
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
                write!(
                    f,
                    "seam: a reserved field carried a nonzero byte (must be zero)"
                )
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
    /// remote `SeamHello` MUST pass this before [`negotiate_with_role`] consumes it.
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

/// The highest profile both validated envelopes advertise (§3.2), or the typed hard failure.
fn select_profile(local: &SeamHello, remote: &SeamHello) -> Result<u16, SeamError> {
    let lp = local.validate()?;
    let rp = remote.validate()?;
    // Both lists ascending; iterate local high→low and take the first that the
    // remote also advertises = max of the intersection.
    lp.iter()
        .rev()
        .find(|p| rp.contains(p))
        .copied()
        .ok_or_else(|| SeamError::VersionMismatch {
            local: lp.to_vec(),
            remote: rp.to_vec(),
        })
}

/// §3.2 and KISS-ANNOUNCE-7.3-0003 (CireSnave decision 310, Q3 A): the highest mutually-supported
/// profile, and a **role-aware** live capability word,
/// `(local & remote & !FEAT) | (provider & FEAT)` with `FEAT` = [`SEAM_CAP_FEAT_MASK`].
///
/// `local_role` is the role of the LOCAL envelope; the provider is known from the exchange, never
/// from a capability bit. A consumer writes the provider-only bits as zero (KISS-ANNOUNCE-7.3-0002),
/// which is why a plain AND would lose them. The FEAT axis is masked as a whole, not bit by bit, so two readers that
/// recognise different FEAT bits still compute the same word. EXT and SUB bits are live only when
/// both sides set them. Both envelopes are validated first; disjoint advertised profile sets hard-fail with
/// [`SeamError::VersionMismatch`], never a guessed version.
pub fn negotiate_with_role(
    local: &SeamHello,
    remote: &SeamHello,
    local_role: Role,
) -> Result<Negotiated, SeamError> {
    let profile = select_profile(local, remote)?;
    let provider_word = match local_role {
        Role::Provider => local.capabilities,
        Role::Consumer => remote.capabilities,
    };
    Ok(Negotiated {
        profile,
        capabilities: (local.capabilities & remote.capabilities & !SEAM_CAP_PROVIDER_ONLY_MASK)
            | (provider_word & SEAM_CAP_PROVIDER_ONLY_MASK),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn seamhello_layout_is_frozen() {
        // The envelope must outlive every profile it negotiates, so its size +
        // field offsets are frozen forever and cross-checked on both sides.
        assert_eq!(
            size_of::<SeamHello>(),
            56,
            "SeamHello is frozen at 56 bytes"
        );
        assert_eq!(
            align_of::<SeamHello>(),
            8,
            "u64 capabilities forces 8-byte align"
        );
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
        // are the EXPLICIT `reserved1` field, so a struct-as-wire envelope can
        // never carry uninitialized garbage a foreign reader rejects:
        // advertise() MUST zero them. Asserted structurally rather than by
        // viewing the struct's bytes through a raw pointer (that view read
        // every byte of the struct as `u8`, which is only sound if there is no
        // implicit padding — and proving that is exactly what the layout
        // assertions below do, with no `unsafe`).
        let hello = SeamHello::fuel(SEAM_CAP_FDX_V1);
        assert_eq!(std::mem::offset_of!(SeamHello, profiles), 10);
        assert_eq!(
            std::mem::offset_of!(SeamHello, reserved1),
            42,
            "reserved1 must start exactly where `profiles` ends"
        );
        assert_eq!(
            std::mem::offset_of!(SeamHello, capabilities),
            42 + std::mem::size_of_val(&hello.reserved1),
            "no implicit padding between reserved1 and capabilities"
        );
        assert_eq!(std::mem::size_of::<SeamHello>(), 56, "no trailing padding");
        assert_eq!(
            hello.reserved1, [0u8; 6],
            "reserved1 alignment padding (bytes 42..48) must be all-zero"
        );
    }

    #[test]
    fn v1_both_sides_select_profile_1() {
        let local = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_FDX_GGML);
        let remote = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_FDX_MX);
        let n = negotiate_with_role(&local, &remote, Role::Consumer).expect("v1 ∩ v1 negotiates");
        assert_eq!(n.profile, PROFILE_V1);
        // capabilities = local & remote — only the common FDX_V1 bit survives.
        assert_eq!(n.capabilities, SEAM_CAP_FDX_V1);
    }

    #[test]
    fn highest_mutually_supported_wins() {
        let local = SeamHello::advertise(&[1, 2, 3], 0);
        let remote = SeamHello::advertise(&[2, 3, 5], 0);
        assert_eq!(
            negotiate_with_role(&local, &remote, Role::Consumer)
                .unwrap()
                .profile,
            3
        );
    }

    #[test]
    fn disjoint_profiles_hard_fail() {
        let local = SeamHello::advertise(&[1], 0);
        let remote = SeamHello::advertise(&[2], 0);
        match negotiate_with_role(&local, &remote, Role::Consumer) {
            Err(SeamError::VersionMismatch { local, remote }) => {
                assert_eq!(local, vec![1]);
                assert_eq!(remote, vec![2]);
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_bad_magic_and_unsorted() {
        let mut bad = SeamHello::fuel(0);
        bad.magic = 0xDEAD_BEEF;
        assert!(matches!(
            bad.validate(),
            Err(SeamError::BadMagic(0xDEAD_BEEF))
        ));

        let mut unsorted = SeamHello::advertise(&[1, 2], 0);
        unsorted.profiles[0] = 2;
        unsorted.profiles[1] = 1; // now descending
        assert!(matches!(unsorted.validate(), Err(SeamError::NotAscending)));

        let mut toomany = SeamHello::fuel(0);
        toomany.profiles_len = (SEAM_MAX_PROFILES + 1) as u16;
        assert!(matches!(
            toomany.validate(),
            Err(SeamError::TooManyProfiles(_))
        ));
    }

    /// A 56-byte frame as the KISS corpus writes it (`conformance/corpus/announce_vectors.json`).
    fn hello_from_hex(hex: &str) -> SeamHello {
        let b: Vec<u8> = hex
            .split_whitespace()
            .map(|x| u8::from_str_radix(x, 16).expect("hex byte"))
            .collect();
        assert_eq!(b.len(), 56, "an envelope is 56 bytes");
        let u16_at = |i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        let mut profiles = [0u16; SEAM_MAX_PROFILES];
        for (k, p) in profiles.iter_mut().enumerate() {
            *p = u16_at(10 + 2 * k);
        }
        SeamHello {
            magic: u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            envelope_version: b[4],
            reserved: [b[5], b[6], b[7]],
            profiles_len: u16_at(8),
            profiles,
            reserved1: [b[42], b[43], b[44], b[45], b[46], b[47]],
            capabilities: u64::from_le_bytes(b[48..56].try_into().unwrap()),
        }
    }

    const KISS_CONSUMER: &str = "53 45 41 4D 01 00 00 00 01 00 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 00 00 00 31 00";
    const KISS_PROVIDER: &str = "53 45 41 4D 01 00 00 00 02 00 01 00 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 03 00 00 00 03 00 91 00";
    /// `live_capabilities_le_hex` of both vectors: `01 00 00 00 03 00 11 00`.
    const KISS_LIVE: u64 = 0x0011_0003_0000_0001;

    /// KISS-ANNOUNCE-7.3-0003, worked vectors (the corpus bytes, both orientations): the live word is
    /// the same, and the plain AND (`0x0011_0000_0000_0001`) loses both FEAT bits.
    #[test]
    fn kiss_vectors_live_word_is_role_aware_in_both_orientations() {
        let consumer = hello_from_hex(KISS_CONSUMER);
        let provider = hello_from_hex(KISS_PROVIDER);
        let a = negotiate_with_role(&consumer, &provider, Role::Consumer).expect("consumer local");
        let b = negotiate_with_role(&provider, &consumer, Role::Provider).expect("provider local");
        assert_eq!(a.profile, 1);
        assert_eq!(b.profile, 1);
        assert_eq!(a.capabilities, KISS_LIVE, "consumer local, provider remote");
        assert_eq!(b.capabilities, KISS_LIVE, "provider local, consumer remote");
        // A plain AND of the two words would lose both FEAT bits.
        let plain = consumer.capabilities & provider.capabilities;
        assert_eq!(plain, 0x0011_0000_0000_0001);
        assert_ne!(plain, KISS_LIVE);
    }

    /// The old JIT test, FLIPPED: the provider flags PROVISION_ON_REQUEST and Fuel (the consumer)
    /// does not write the provider-only bit (7.3-0002), and the feature is ON.
    #[test]
    fn a_provider_only_bit_is_live_when_only_the_provider_sets_it() {
        let fuel = SeamHello::fuel(SEAM_CAP_FDX_V1);
        let provider = SeamHello::fuel(
            SEAM_CAP_FDX_V1 | SEAM_CAP_PROVISION_ON_REQUEST | SEAM_CAP_CONTRACT_QUERY,
        );
        let n = negotiate_with_role(&fuel, &provider, Role::Consumer).unwrap();
        assert_eq!(
            n.capabilities & SEAM_CAP_PROVISION_ON_REQUEST,
            SEAM_CAP_PROVISION_ON_REQUEST
        );
        assert_eq!(
            n.capabilities & SEAM_CAP_CONTRACT_QUERY,
            SEAM_CAP_CONTRACT_QUERY
        );
        assert_eq!(n.capabilities & SEAM_CAP_FDX_V1, SEAM_CAP_FDX_V1);
    }

    /// SABOTAGE SIBLING of the test above: a provider-only bit that ONLY the consumer sets is never
    /// live (only the provider's word decides a FEAT bit), so a consumer cannot switch a provider
    /// feature on by claiming it. Without it the flip above could pass for the wrong reason (a
    /// formula that took FEAT from `local | remote` would also pass it).
    #[test]
    fn a_provider_only_bit_set_only_by_the_consumer_is_never_live() {
        let fuel = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_PROVISION_ON_REQUEST);
        let provider = SeamHello::fuel(SEAM_CAP_FDX_V1);
        let as_consumer = negotiate_with_role(&fuel, &provider, Role::Consumer).unwrap();
        assert_eq!(as_consumer.capabilities & SEAM_CAP_PROVISION_ON_REQUEST, 0);
        // And with the roles exchanged (the claiming side is now the provider's remote).
        let as_provider = negotiate_with_role(&provider, &fuel, Role::Provider).unwrap();
        assert_eq!(as_provider.capabilities & SEAM_CAP_PROVISION_ON_REQUEST, 0);
    }

    /// EXT and SUB bits stay a plain AND: live only if BOTH sides set them.
    #[test]
    fn ext_and_sub_bits_are_live_only_when_both_sides_set_them() {
        let sub = 1u64 << 52;
        let consumer = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_FDX_MX | sub);
        let provider = SeamHello::fuel(SEAM_CAP_FDX_V1 | SEAM_CAP_FDX_GGML);
        let n = negotiate_with_role(&consumer, &provider, Role::Consumer).unwrap();
        assert_eq!(n.capabilities, SEAM_CAP_FDX_V1);
    }

    /// 7.3-0003 masks by AXIS, not by recognition: a FEAT bit this crate has never heard of
    /// (bit 35, core-reserved) is still decided by the provider alone, so two implementations that
    /// recognise different bits compute the same word.
    #[test]
    fn an_unrecognised_feat_bit_follows_the_provider_too() {
        let unknown_feat = 1u64 << 35;
        let consumer = SeamHello::fuel(SEAM_CAP_FDX_V1);
        let provider = SeamHello::fuel(SEAM_CAP_FDX_V1 | unknown_feat);
        let n = negotiate_with_role(&consumer, &provider, Role::Consumer).unwrap();
        assert_eq!(n.capabilities & unknown_feat, unknown_feat);
    }

    /// The profile rule is unchanged: highest mutual wins, a disjoint set is a typed hard failure.
    #[test]
    fn role_aware_negotiation_keeps_the_profile_rules() {
        let local = SeamHello::advertise(&[1, 2, 3], 0);
        let remote = SeamHello::advertise(&[2, 3, 5], 0);
        assert_eq!(
            negotiate_with_role(&local, &remote, Role::Consumer)
                .unwrap()
                .profile,
            3
        );
        let a = SeamHello::advertise(&[1], 0);
        let b = SeamHello::advertise(&[2], 0);
        assert!(matches!(
            negotiate_with_role(&a, &b, Role::Provider),
            Err(SeamError::VersionMismatch { .. })
        ));
    }

    /// The axis masks partition the word and the registry bits sit inside the right axis.
    #[test]
    fn the_axis_masks_partition_the_word_and_hold_the_registry_bits() {
        assert_eq!(
            SEAM_CAP_EXT_MASK | SEAM_CAP_FEAT_MASK | SEAM_CAP_SUB_MASK,
            u64::MAX
        );
        assert_eq!(SEAM_CAP_EXT_MASK & SEAM_CAP_FEAT_MASK, 0);
        assert_eq!(SEAM_CAP_FEAT_MASK & SEAM_CAP_SUB_MASK, 0);
        assert_eq!(SEAM_CAP_PROVIDER_ONLY_MASK, SEAM_CAP_FEAT_MASK);
        assert_ne!(SEAM_CAP_PROVISION_ON_REQUEST & SEAM_CAP_FEAT_MASK, 0);
        assert_ne!(SEAM_CAP_CONTRACT_QUERY & SEAM_CAP_FEAT_MASK, 0);
        assert_eq!(SEAM_CAP_PROVISION_ON_REQUEST.trailing_zeros(), 32);
        assert_eq!(SEAM_CAP_CONTRACT_QUERY.trailing_zeros(), 33);
        for ext in [
            SEAM_CAP_FDX_V1,
            SEAM_CAP_FDX_MX,
            SEAM_CAP_FDX_GGML,
            SEAM_CAP_FDX_AFFINE,
            SEAM_CAP_FDX_SYMBOLIC,
            SEAM_CAP_FDX_GATHER,
        ] {
            assert_ne!(ext & SEAM_CAP_EXT_MASK, 0);
        }
    }
}
