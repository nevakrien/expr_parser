use crate::type_kinds::{KindId, MutId, PtrId};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Projection {
    /// &T -> T
    SimpleDeref,

    /// &s -> &s.x
    FieldReref(u32),

    /// *T -> &T
    RawReref,

    /// &T -> *T
    ForgetSafe,

    /// T -> f(T)
    SmartCall,

    /// T -> S
    Casted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DerefStep {
    pub proj: Projection,
    pub id: KindId,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DerefTo {
    pub parent: KindId,
    pub style: PtrId,
    pub tgt: KindId,
    pub mutable: MutId,
    pub next_in_chain: Option<Option<DerefStep>>,
}
