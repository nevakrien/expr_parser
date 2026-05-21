use super::{KindId, MutId, Projection, PtrId};
use crate::ir::ValId;

#[derive(Debug, PartialEq, Hash)]
pub struct UseUn {
    pub src: KindId,
    pub tgt: KindId,
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
