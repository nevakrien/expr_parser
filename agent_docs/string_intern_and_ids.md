# String Interning and ID Maps

This doc covers related low-level utilities under `src/data_structures/`:

- stable string IDs (`StrId`) via `StringInterner`
- fast ID-keyed hash maps (`IdHashMap`) via `IdentityHasher`

Current file layout:

- `src/data_structures/string_intern.rs`: `StrId`, `StringInterner`, hardcoded names
- `src/data_structures/identity_hasher.rs`: `IdHashMap` and hasher choice
- `src/data_structures/index.rs`: `Idx`, `IndexVec`, `UnionFind`
- `src/data_structures/graph.rs`: graph traits, SCC/topological helpers, basic graph storage

## `StrId` Stability Contract

`StringInterner::new()` pre-interns parser keywords and builtin special member names.

Order is fixed by:

- `parsing::KEYWORDS`
- `EXTRA_HARD_CODED_NAMES`

`get_known_strid("...")` is `const` and assumes this ordering is stable.

Important: changing keyword/extra-name ordering changes hardcoded `StrId` constants (`ADD_STR`, `FREE_STR`, etc.) and can silently break method/operator lookup.

## Interner Storage Model

Memory layout is compact:

- `bytes: Vec<u8>` stores all interned strings back-to-back.
- `spans: Vec<(offset, len)>` maps `StrId -> byte slice`.
- `table: Vec<Entry { hash, id }>` is an open-addressed hash table with linear probing.

`hash == 0` is reserved as empty sentinel; `scrub_hash` remaps real hash `0 -> 1`.

## Interner Operations

- `intern(&str) -> StrId`
  - hash input
  - probe table for matching hash+bytes
  - return existing id or append new bytes/span + insert entry
- `resolve(StrId) -> &str`
  - reads `(offset, len)` and returns slice as UTF-8

Growth policy:

- table grows x4 when load exceeds 70%
- `bytes` and `spans` capacity also grow by factor 4 when needed

## Safety Assumptions

`resolve` uses `from_utf8_unchecked`.

This is safe under the module invariant that all interned input comes from valid Rust `&str` and bytes are never mutated after insertion except by append.

## `IdHashMap` and `IdentityHasher`

`IdHashMap<K, V>` is a `HashMap` using `BuildHasherDefault<IdentityHasher>`.

Hasher behavior:

- numeric `write_*` mostly forwards value directly (`u64` is identity)
- generic byte writes use FNV-1a style fallback

This is optimized for compiler-internal ID keys (`NameId`, `TypeId`, etc.) where keys are already uniformly assigned and hashing overhead should stay minimal.

## Practical Guidance

- Use `IdHashMap` for dense integer-like IDs and interned symbol IDs.
- Avoid using this hasher for attacker-controlled hash-map keys in network-facing contexts.
- If you add new builtin special names, add them to `EXTRA_HARD_CODED_NAMES` and keep docs/tests in sync with the known-id contract.
