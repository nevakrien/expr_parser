use crate::parsing::KEYWORDS;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StrId(usize);

const DEFAULT_BUCKETS: usize = 2048;
const MAX_LOAD_NUMERATOR: usize = 7;
const MAX_LOAD_DENOMINATOR: usize = 10;
const DEFAULT_BYTES_CAPACITY: usize = 16 * 1024;
const DEFAULT_SPANS_CAPACITY: usize = 1024;
const BYTES_GROWTH_FACTOR: usize = 4;
const SPANS_GROWTH_FACTOR: usize = 4;

#[derive(Debug, Copy, Clone)]
struct Entry {
    hash: u64,
    id: StrId,
}

pub const EXTRA_HARD_CODED_NAMES: &[&str] = &[
    "static",
    "raw",
    "__free",
    "__size_of",
    "__align_of",
    "__user_free",
    "__deref",
    "__deref_mut",
    "__add",
    "__sub",
    "__mul",
    "__div",
    "__mod",
    "__bitand",
    "__bitor",
    "__bitxor",
    "__shl",
    "__shr",
    "__eq",
    "__ne",
    "__lt",
    "__le",
    "__gt",
    "__ge",
    "__neg",
    "__not",
    "__bitnot",
    "__pre_inc",
    "__post_inc",
    "__pre_dec",
    "__post_dec",
];

const fn const_str_eq(a: &str, b: &str) -> bool {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();

    if a_bytes.len() != b_bytes.len() {
        return false;
    }

    let mut i = 0;
    while i < a_bytes.len() {
        if a_bytes[i] != b_bytes[i] {
            return false;
        }
        i += 1;
    }

    true
}

//hard rely on this
pub const fn get_known_strid(s: &str) -> StrId {
    let mut i = 0;

    while i < KEYWORDS.len() {
        if const_str_eq(KEYWORDS[i], s) {
            return StrId(i);
        }
        i += 1;
    }

    let mut j = 0;
    while j < EXTRA_HARD_CODED_NAMES.len() {
        if const_str_eq(EXTRA_HARD_CODED_NAMES[j], s) {
            return StrId(KEYWORDS.len() + j);
        }
        j += 1;
    }

    panic!("not a known keyword");
}

#[test]
fn known_stuff_works() {
    let mut intern = StringInterner::new();
    for k in KEYWORDS.iter().chain(EXTRA_HARD_CODED_NAMES.iter()) {
        let i = intern.intern(k);
        assert_eq!(i, get_known_strid(k));
        assert_eq!(*k, intern.resolve(i));
    }
}

pub const STATIC_STR: StrId = get_known_strid("static");
pub const RAW_STR: StrId = get_known_strid("raw");
pub const ADD_STR: StrId = get_known_strid("__add");
pub const SUB_STR: StrId = get_known_strid("__sub");
pub const MUL_STR: StrId = get_known_strid("__mul");
pub const DIV_STR: StrId = get_known_strid("__div");
pub const MOD_STR: StrId = get_known_strid("__mod");
pub const BITAND_STR: StrId = get_known_strid("__bitand");
pub const BITOR_STR: StrId = get_known_strid("__bitor");
pub const BITXOR_STR: StrId = get_known_strid("__bitxor");
pub const SHL_STR: StrId = get_known_strid("__shl");
pub const SHR_STR: StrId = get_known_strid("__shr");
pub const EQ_STR: StrId = get_known_strid("__eq");
pub const NE_STR: StrId = get_known_strid("__ne");
pub const LT_STR: StrId = get_known_strid("__lt");
pub const LE_STR: StrId = get_known_strid("__le");
pub const GT_STR: StrId = get_known_strid("__gt");
pub const GE_STR: StrId = get_known_strid("__ge");
pub const NEG_STR: StrId = get_known_strid("__neg");
pub const NOT_STR: StrId = get_known_strid("__not");
pub const BITNOT_STR: StrId = get_known_strid("__bitnot");
pub const PRE_INC_STR: StrId = get_known_strid("__pre_inc");
pub const POST_INC_STR: StrId = get_known_strid("__post_inc");
pub const PRE_DEC_STR: StrId = get_known_strid("__pre_dec");
pub const POST_DEC_STR: StrId = get_known_strid("__post_dec");
pub const FREE_STR: StrId = get_known_strid("__free");
pub const SIZE_OF_STR: StrId = get_known_strid("__size_of");
pub const ALIGN_OF_STR: StrId = get_known_strid("__align_of");
pub const USER_FREE_STR: StrId = get_known_strid("__user_free");
pub const DEREF_STR: StrId = get_known_strid("__deref");
pub const DEREF_MUT_STR: StrId = get_known_strid("__deref_mut");

#[derive(Debug)]
pub struct StringInterner {
    /// All string bytes, tightly packed
    bytes: Vec<u8>,

    /// StrId -> (offset, len)
    spans: Vec<(usize, usize)>,

    /// hash table of (hash, StrId) with linear probing; hash == 0 means empty
    table: Vec<Entry>,
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl StringInterner {
    pub fn new() -> Self {
        Self::with_buckets(DEFAULT_BUCKETS)
    }

    pub fn with_buckets(bucket_count: usize) -> Self {
        let mut ans = Self::__no_defualts_with_buckets(bucket_count);

        for k in KEYWORDS.iter().chain(EXTRA_HARD_CODED_NAMES.iter()) {
            ans.intern(k);
        }

        ans
    }

    fn __no_defualts_with_buckets(bucket_count: usize) -> Self {
        assert!(bucket_count.is_power_of_two());

        Self {
            bytes: Vec::with_capacity(DEFAULT_BYTES_CAPACITY),
            spans: Vec::with_capacity(DEFAULT_SPANS_CAPACITY),
            table: vec![
                Entry {
                    hash: 0,
                    id: StrId(0)
                };
                bucket_count
            ],
        }
    }

    /// Intern a string slice.
    /// If an equal string was interned before, returns the same StrId.
    #[inline(always)]
    pub fn intern(&mut self, s: &str) -> StrId {
        let bytes = s.as_bytes();
        let h = scrub_hash(hash_bytes(bytes));

        let (idx, found) = self.find_slot(h, bytes);
        if found {
            return self.table[idx].id;
        }

        self.insert_slow_path(idx, h, bytes)
    }

    #[inline(always)]
    pub fn resolve(&self, id: StrId) -> &str {
        let (off, len) = self.spans[id.0];
        unsafe { std::str::from_utf8_unchecked(&self.bytes[off..off + len]) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        hash as usize & (self.table.len() - 1)
    }

    #[inline(always)]
    fn maybe_grow(&mut self) {
        if self.table.is_empty() {
            return;
        }

        if self.spans.len() * MAX_LOAD_DENOMINATOR <= self.table.len() * MAX_LOAD_NUMERATOR {
            return;
        }
        self.grow();
    }

    #[cold]
    fn grow(&mut self) {
        let new_bucket_count = self.table.len() * 4;
        let old_table = std::mem::replace(
            &mut self.table,
            vec![
                Entry {
                    hash: 0,
                    id: StrId(0)
                };
                new_bucket_count
            ],
        );

        for entry in old_table {
            if entry.hash != 0 {
                self.insert_entry(entry.hash, entry.id);
            }
        }
    }

    fn insert_slow_path(&mut self, idx: usize, hash: u64, bytes: &[u8]) -> StrId {
        let id = StrId(self.spans.len());
        let off = self.bytes.len();
        let len = bytes.len();

        self.ensure_bytes_capacity(len);
        self.bytes.extend_from_slice(bytes);
        self.ensure_spans_capacity(1);
        self.spans.push((off, len));
        self.table[idx] = Entry { hash, id };
        self.maybe_grow();

        id
    }

    #[inline(always)]
    fn ensure_bytes_capacity(&mut self, additional: usize) {
        let needed = self.bytes.len() + additional;
        if needed <= self.bytes.capacity() {
            return;
        }

        let mut new_cap = self.bytes.capacity().max(1);
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(BYTES_GROWTH_FACTOR);
        }
        self.bytes.reserve_exact(new_cap - self.bytes.capacity());
    }

    #[inline(always)]
    fn ensure_spans_capacity(&mut self, additional: usize) {
        let needed = self.spans.len() + additional;
        if needed <= self.spans.capacity() {
            return;
        }

        let mut new_cap = self.spans.capacity().max(1);
        while new_cap < needed {
            new_cap = new_cap.saturating_mul(SPANS_GROWTH_FACTOR);
        }
        self.spans.reserve_exact(new_cap - self.spans.capacity());
    }

    #[inline(always)]
    fn find_slot(&self, hash: u64, bytes: &[u8]) -> (usize, bool) {
        let mut idx = self.bucket_index(hash);
        let mask = self.table.len() - 1;

        loop {
            let entry = self.table[idx];
            if entry.hash == 0 {
                return (idx, false);
            }
            if entry.hash == hash {
                let (off, len) = self.spans[entry.id.0];
                if &self.bytes[off..off + len] == bytes {
                    return (idx, true);
                }
            }
            idx = (idx + 1) & mask;
        }
    }

    #[inline(always)]
    fn insert_entry(&mut self, hash: u64, id: StrId) {
        let mut idx = self.bucket_index(hash);
        let mask = self.table.len() - 1;

        loop {
            if self.table[idx].hash == 0 {
                self.table[idx] = Entry { hash, id };
                return;
            }
            idx = (idx + 1) & mask;
        }
    }
}

#[inline]
fn hash_bytes(bytes: &[u8]) -> u64 {
    // simple FNV-1a style hash (fast, non-cryptographic)
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[inline]
fn scrub_hash(h: u64) -> u64 {
    if h == 0 { 1 } else { h }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_string_same_id() {
        let mut i = StringInterner::__no_defualts_with_buckets(DEFAULT_BUCKETS);

        let a = i.intern("hello");
        let b = i.intern("hello");

        assert_eq!(a, b);
        assert_eq!(i.len(), 1);
        assert_eq!(i.resolve(a), "hello");
    }

    #[test]
    fn different_strings_different_ids() {
        let mut i = StringInterner::__no_defualts_with_buckets(DEFAULT_BUCKETS);

        let a = i.intern("hello");
        let b = i.intern("world");

        assert_ne!(a, b);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn forced_single_bucket_collision() {
        let mut i = StringInterner::__no_defualts_with_buckets(1);

        let a = i.intern("foo");
        let b = i.intern("bar");
        let c = i.intern("baz");
        let d = i.intern("foo");

        assert_eq!(a, d);
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_eq!(i.len(), 3);
    }

    #[test]
    fn growth_preserves_ids() {
        let mut i = StringInterner::__no_defualts_with_buckets(1);

        let mut ids = Vec::new();
        for n in 0..100 {
            ids.push(i.intern(&format!("sym_{n}")));
        }

        for (n, id) in ids.iter().enumerate() {
            let id2 = i.intern(&format!("sym_{n}"));
            assert_eq!(*id, id2);
        }

        assert_eq!(i.len(), 100);
    }
}
