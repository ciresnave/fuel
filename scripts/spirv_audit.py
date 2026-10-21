# SPDX-License-Identifier: MIT OR Apache-2.0
"""Static audit of the committed Vulkan SPIR-V against the Vulkan feature rules the
Khronos validation layer reports at pipeline creation.

Usage: python scripts/spirv_audit.py [spv_dir] [--markdown]

For every `*.spv` in `fuel-vulkan-kernels/spv` it reports, without a GPU:

* the 16-bit storage capabilities the module declares
  (`UniformAndStorageBuffer16BitAccess` needs the Vulkan 1.1 feature
  `uniformAndStorageBuffer16BitAccess`; VUID-VkShaderModuleCreateInfo-pCode-08740 and
  VUID-RuntimeSpirv-uniformAndStorageBuffer16BitAccess-06332);
* 64-bit integer atomics on buffers (VUID-RuntimeSpirv-None-06278, feature
  `shaderBufferInt64Atomics`);
* workgroup (shared) memory above 32,768 bytes (VUID-RuntimeSpirv-Workgroup-06530
  when the device limit `maxComputeSharedMemorySize` is 32,768);
* GLSL.std.450 extended instructions applied to 64-bit floats, which the Vulkan
  environment does not allow (VUID-VkShaderModuleCreateInfo-pCode-08737, `spirv-val`:
  "expected Result Type to be a 16 or 32-bit scalar or vector float type").

It is a parser of the binary module (opcodes, types, constants, variables), not a
validator: it answers "which committed shaders fall in each class", and the counts
should be cross-checked against a `validate_sync` run of the layer.
"""

import struct
import sys
from pathlib import Path

CAP = {
    9: "Float16",
    10: "Float64",
    11: "Int64",
    12: "Int64Atomics",
    22: "Int16",
    5345: "VulkanMemoryModel",
    4433: "StorageBuffer16BitAccess",
    4434: "UniformAndStorageBuffer16BitAccess",
    4435: "StoragePushConstant16",
    4436: "StorageInputOutput16",
}
ATOMIC_OPS = set(range(227, 243)) - {228}  # OpAtomicLoad..OpAtomicXor, minus Store (no result type)
GLSL450_NAMES = {27: "Exp", 28: "Log", 29: "Exp2", 30: "Log2", 26: "Pow", 13: "Sin", 14: "Cos", 15: "Tan",
                 16: "Asin", 17: "Acos", 18: "Atan", 19: "Sinh", 20: "Cosh", 21: "Tanh", 22: "Asinh",
                 23: "Acosh", 24: "Atanh", 25: "Atan2"}
# GLSL.std.450 instructions whose Result Type must be a 16- or 32-bit float.
FLOAT_32_ONLY = set(GLSL450_NAMES)


def parse(path):
    data = path.read_bytes()
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    assert words[0] == 0x07230203, f"{path.name}: not SPIR-V"
    i = 5
    caps, types, consts, variables, ext_sets = [], {}, {}, [], {}
    atomics_64, glsl_f64 = [], []
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        a = words[i + 1:i + wc]
        if op == 17:
            caps.append(a[0])
        elif op == 11:
            raw = b"".join(struct.pack("<I", w) for w in a[1:]).split(b"\0")[0].decode()
            ext_sets[a[0]] = raw
        elif op == 21:
            types[a[0]] = ("int", a[1])
        elif op == 22:
            types[a[0]] = ("float", a[1])
        elif op == 23:
            types[a[0]] = ("vec", a[1], a[2])
        elif op == 24:
            types[a[0]] = ("mat", a[1], a[2])
        elif op == 28:
            types[a[0]] = ("array", a[1], a[2])
        elif op == 29:
            types[a[0]] = ("rtarray", a[1])
        elif op == 30:
            types[a[0]] = ("struct", list(a[1:]))
        elif op == 32:
            types[a[0]] = ("ptr", a[1], a[2])
        elif op == 43:
            consts[a[1]] = a[2]
        elif op == 59:
            variables.append((a[0], a[1], a[2]))  # (type id, id, storage class)
        elif op in ATOMIC_OPS and wc > 3:
            t = types.get(a[0])
            if t and t[0] == "int" and t[1] == 64:
                atomics_64.append(op)
        elif op == 12 and wc > 4:  # OpExtInst: result type, result id, set, instruction
            if ext_sets.get(a[2]) == "GLSL.std.450" and a[3] in FLOAT_32_ONLY:
                t = types.get(a[0])
                while t and t[0] == "vec":
                    t = types.get(t[1])
                if t and t[0] == "float" and t[1] == 64:
                    glsl_f64.append(GLSL450_NAMES[a[3]])
        i += wc
    return caps, types, consts, variables, atomics_64, glsl_f64


def size_of(types, consts, tid):
    t = types[tid]
    if t[0] in ("int", "float"):
        return t[1] // 8
    if t[0] == "vec":
        return size_of(types, consts, t[1]) * t[2]
    if t[0] == "mat":
        return size_of(types, consts, t[1]) * t[2]
    if t[0] == "array":
        return size_of(types, consts, t[1]) * consts[t[2]]
    if t[0] == "struct":
        return sum(size_of(types, consts, m) for m in t[1])
    return 0


def audit(path):
    caps, types, consts, variables, atomics_64, glsl_f64 = parse(path)
    shared = 0
    for ptype, _vid, sc in variables:
        if sc == 4:  # Workgroup
            p = types[ptype]
            shared += size_of(types, consts, p[2])
    return {
        "name": path.stem,
        "caps16": [CAP[c] for c in caps if c in (4433, 4434, 4435, 4436)],
        "uni16": 4434 in caps,
        "int16": 22 in caps,
        "vmm": 5345 in caps,
        "atomics64": bool(atomics_64) or 12 in caps,
        "shared": shared,
        "glsl_f64": sorted(set(glsl_f64)),
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--markdown" in sys.argv
    root = Path(args[0]) if args else Path(__file__).resolve().parent.parent / "fuel-vulkan-kernels" / "spv"
    rows = [audit(p) for p in sorted(root.glob("*.spv"))]
    classes = {
        "uniformAndStorageBuffer16BitAccess (08740 + 06332)": [r["name"] for r in rows if r["uni16"]],
        "shaderInt16 (08740, Int16 capability)": [r["name"] for r in rows if r["int16"]],
        "vulkanMemoryModel (08740)": [r["name"] for r in rows if r["vmm"]],
        "shaderBufferInt64Atomics (06278)": [r["name"] for r in rows if r["atomics64"]],
        "shared memory > 32768 B (06530)": [f'{r["name"]} ({r["shared"]} B)' for r in rows if r["shared"] > 32768],
        "GLSL.std.450 on f64 (08737)": [f'{r["name"]} ({", ".join(r["glsl_f64"])})' for r in rows if r["glsl_f64"]],
    }
    print(f"{len(rows)} modules in {root}")
    for k, v in classes.items():
        print(f"\n{k}: {len(v)}")
        for name in v:
            print(f"  {name}")
    if md:
        print("\n| shader | classes |\n|---|---|")
        for r in rows:
            c = []
            if r["uni16"]:
                c.append("uniformAndStorageBuffer16BitAccess")
            if r["int16"]:
                c.append("shaderInt16")
            if r["vmm"]:
                c.append("vulkanMemoryModel")
            if r["atomics64"]:
                c.append("shaderBufferInt64Atomics")
            if r["shared"] > 32768:
                c.append(f'shared {r["shared"]} B')
            if r["glsl_f64"]:
                c.append("f64 " + "/".join(r["glsl_f64"]))
            if c:
                print(f'| `{r["name"]}` | {", ".join(c)} |')


if __name__ == "__main__":
    main()
