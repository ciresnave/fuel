//! imatrix (importance matrix) parser — llama.cpp activation
//! statistics for quantization calibration.
//!
//! Wire format (little-endian throughout):
//! - `i32` n_entries
//! - per entry:
//!   - `i32` name length, then UTF-8 bytes
//!   - `i32` ncall — number of forward passes that contributed
//!   - `i32` nval — number of f32 values
//!   - `nval × f32` activation magnitudes
//!
//! Returned values are normalized — divided by `ncall` when non-zero,
//! returned raw when `ncall == 0`. This matches llama.cpp's
//! `imatrix.cpp` consumer behaviour.
//!
//! The parser is transport-independent: hand it any `impl Read` (file,
//! `Cursor<&[u8]>`, network stream, decompressor). Convenience wrappers
//! for `&[u8]` and file paths are provided.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use fuel_core_types::{Error, Result, bail};

/// Parse an imatrix stream from any source.
///
/// Returns `name -> normalized activations` for every entry in the
/// stream.
pub fn parse<R: Read>(reader: &mut R) -> Result<HashMap<String, Vec<f32>>> {
    let n_entries = reader
        .read_i32::<LittleEndian>()
        .map_err(|e| Error::msg(format!("failed to read number of entries: {e}")))?
        as usize;

    if n_entries < 1 {
        bail!("imatrix has no entries");
    }

    let mut out = HashMap::with_capacity(n_entries);
    for i in 0..n_entries {
        let len = reader
            .read_i32::<LittleEndian>()
            .map_err(|e| Error::msg(format!("entry {}: failed to read name length: {e}", i + 1)))?
            as usize;

        let mut name_buf = vec![0u8; len];
        reader
            .read_exact(&mut name_buf)
            .map_err(|e| Error::msg(format!("entry {}: failed to read name: {e}", i + 1)))?;
        let name = String::from_utf8(name_buf)
            .map_err(|e| Error::msg(format!("entry {}: invalid UTF-8 name: {e}", i + 1)))?;

        let ncall = reader
            .read_i32::<LittleEndian>()
            .map_err(|e| Error::msg(format!("entry {}: failed to read ncall: {e}", i + 1)))?
            as usize;
        let nval = reader
            .read_i32::<LittleEndian>()
            .map_err(|e| Error::msg(format!("entry {}: failed to read nval: {e}", i + 1)))?
            as usize;

        if nval < 1 {
            bail!("entry {}: invalid nval {}", i + 1, nval);
        }

        let mut data = Vec::with_capacity(nval);
        for j in 0..nval {
            let v = reader.read_f32::<LittleEndian>().map_err(|e| {
                Error::msg(format!(
                    "entry {} ({}): failed to read value {}: {e}",
                    i + 1,
                    name,
                    j
                ))
            })?;
            data.push(if ncall == 0 { v } else { v / ncall as f32 });
        }
        out.insert(name, data);
    }

    Ok(out)
}

/// Parse an imatrix from an in-memory byte buffer.
pub fn parse_bytes(bytes: &[u8]) -> Result<HashMap<String, Vec<f32>>> {
    let mut cursor = Cursor::new(bytes);
    parse(&mut cursor)
}

/// Open and parse an imatrix file from disk.
pub fn load_path<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Vec<f32>>> {
    let path = path.as_ref();
    let file = File::open(path)
        .map_err(|e| Error::msg(format!("failed to open {}: {e}", path.display())))?;
    let mut reader = BufReader::new(file);
    parse(&mut reader).map_err(|e| e.with_path(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::WriteBytesExt;

    fn write_entry(buf: &mut Vec<u8>, name: &str, ncall: i32, values: &[f32]) {
        buf.write_i32::<LittleEndian>(name.len() as i32).unwrap();
        buf.extend_from_slice(name.as_bytes());
        buf.write_i32::<LittleEndian>(ncall).unwrap();
        buf.write_i32::<LittleEndian>(values.len() as i32).unwrap();
        for v in values {
            buf.write_f32::<LittleEndian>(*v).unwrap();
        }
    }

    #[test]
    fn round_trip_two_entries_normalized() {
        let mut buf = Vec::new();
        buf.write_i32::<LittleEndian>(2).unwrap();
        write_entry(&mut buf, "layer.0", 4, &[8.0, 16.0]);
        write_entry(&mut buf, "layer.1", 0, &[3.0, -3.0]);

        let parsed = parse_bytes(&buf).unwrap();
        assert_eq!(parsed["layer.0"], vec![2.0, 4.0]); // divided by ncall=4
        assert_eq!(parsed["layer.1"], vec![3.0, -3.0]); // ncall=0 → raw
    }

    #[test]
    fn empty_file_is_rejected() {
        let mut buf = Vec::new();
        buf.write_i32::<LittleEndian>(0).unwrap();
        assert!(parse_bytes(&buf).is_err());
    }
}
