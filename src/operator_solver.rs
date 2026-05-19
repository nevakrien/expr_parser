use crate::ir::ValId;
use crate::type_kinds::{KindId, MutId, PtrId};

#[derive(Debug, PartialEq, Hash)]
pub struct UseUn {
    pub src: KindId,
    pub tgt: KindId,
}

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

#[derive(Debug, PartialEq, Hash)]
pub struct DerefTo {
    pub parent: KindId,
    pub style: PtrId,
    pub tgt: KindId,
    pub mutable: MutId,
    pub val: ValId,
}
