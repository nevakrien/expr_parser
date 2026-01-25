#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StrId(usize);

const DEFAULT_BUCKETS: usize = 1024;
const MAX_LOAD_PER_BUCKET: usize = 4;

#[derive(Debug)]
pub struct StringInterner {
    /// All string bytes, tightly packed
    bytes: Vec<u8>,

    /// StrId -> (offset, len)
    spans: Vec<(usize, usize)>,

    /// bucket -> [(StrId, hash)]
    buckets: Vec<Vec<(StrId, u64)>>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::with_buckets(DEFAULT_BUCKETS)
    }

    pub fn with_buckets(bucket_count: usize) -> Self {
        assert!(bucket_count.is_power_of_two());

        Self {
            bytes: Vec::new(),
            spans: Vec::new(),
            buckets: vec![Vec::new(); bucket_count],
        }
    }

    /// Intern a string slice.
    /// If an equal string was interned before, returns the same StrId.
    #[inline]
    pub fn intern(&mut self, s: &str) -> StrId {
        let bytes = s.as_bytes();
        let h = hash_bytes(bytes);
        let bucket = self.bucket_index(h);

        for &(id, existing_h) in &self.buckets[bucket] {
            if existing_h != h {
                continue;
            }
            let (off, len) = self.spans[id.0];
            if &self.bytes[off..off + len] == bytes {
                return id;
            }
        }

        // Insert new string
        let id = StrId(self.spans.len());
        let off = self.bytes.len();
        let len = bytes.len();

        self.bytes.extend_from_slice(bytes);
        self.spans.push((off, len));
        self.buckets[bucket].push((id, h));

        self.maybe_grow();

        id
    }

    #[inline]
    pub fn resolve(&self, id: StrId) -> &str {
        let (off, len) = self.spans[id.0];
        unsafe { std::str::from_utf8_unchecked(&self.bytes[off..off + len]) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        hash as usize & (self.buckets.len() - 1)
    }

    fn maybe_grow(&mut self) {
        if self.spans.len() <= self.buckets.len() * MAX_LOAD_PER_BUCKET {
            return;
        }

        let new_bucket_count = self.buckets.len() * 2;
        let mut new_buckets: Vec<Vec<(StrId, u64)>> =
            vec![Vec::new(); new_bucket_count];

        for bucket in self.buckets.iter() {
            for &(id, h) in bucket {
                let new_index = h as usize & (new_bucket_count - 1);
                new_buckets[new_index].push((id, h));
            }
        }

        self.buckets = new_buckets;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_string_same_id() {
        let mut i = StringInterner::new();

        let a = i.intern("hello");
        let b = i.intern("hello");

        assert_eq!(a, b);
        assert_eq!(i.len(), 1);
        assert_eq!(i.resolve(a), "hello");
    }

    #[test]
    fn different_strings_different_ids() {
        let mut i = StringInterner::new();

        let a = i.intern("hello");
        let b = i.intern("world");

        assert_ne!(a, b);
        assert_eq!(i.len(), 2);
    }

    #[test]
    fn forced_single_bucket_collision() {
        let mut i = StringInterner::with_buckets(1);

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
        let mut i = StringInterner::with_buckets(1);

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
