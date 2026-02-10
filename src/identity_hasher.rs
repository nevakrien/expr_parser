use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Default)]
pub struct IdentityHasher(u64);

// fn hash(x: u64) -> u64 {
//     let full = (x as u128).wrapping_mul(0x243f6a8885a308d3 as u128);
//         let lo = full as u64;
//         let hi = (full >> 64) as u64;
//         lo ^ hi
// }
fn hash(x: u64) -> u64 {x}

impl Hasher for IdentityHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut h = 0xcbf29ce484222325u64;
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        self.0 = h;
    }

    fn write_u8(&mut self, i: u8) {
        self.0 = hash(i as u64);
    }

    fn write_u16(&mut self, i: u16) {
        self.0 = hash(i as u64);
    }

    fn write_u32(&mut self, i: u32) {
        self.0 = hash(i as u64);
    }

    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = hash(i as u64);
    }

    fn write_i8(&mut self, i: i8) {
        self.0 = hash(i as u64);
    }

    fn write_i16(&mut self, i: i16) {
        self.0 = hash(i as u64);
    }

    fn write_i32(&mut self, i: i32) {
        self.0 = hash(i as u64);
    }

    fn write_i64(&mut self, i: i64) {
        self.0 = hash(i as u64);
    }

    fn write_isize(&mut self, i: isize) {
        self.0 = hash(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type IdBuildHasher = BuildHasherDefault<IdentityHasher>;
pub type IdHashMap<K, V> = HashMap<K, V, IdBuildHasher>;
// pub type IdHashMap<K, V> = foldhash::HashMap<K,V>;
