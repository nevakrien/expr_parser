mod errors;
mod kinds;
mod operator_solver;
mod origin;
mod solving;

pub use errors::{TypeClash, TypeError};
pub use kinds::{
    ArraySize, BUILTINS, BuiltinKind, FloatKind, FloatSize, GenId, HARD_CODED_BUILTIN_KINDS,
    IntKind, IntSign, IntSize, KindId, LifeId, LifeKind, MutId, Nullable, PointerStyle, PtrId,
    StructId, TypeKind,
};
pub use operator_solver::{DerefStep, DerefTo, Projection, UseUn};
pub use solving::{
    InnerFunctionTypes, KindLookUp, KindStorage, MutConflict, MutGuess, MutGuessMode, MutInfo,
    MutReason, MutReasonPath, MutSetRes, OriginId, OriginNode, OriginVec, SolvedFunctionTypes,
    SolvedMemberMethodAccessType, SolvedTypes, TypeIntern, TypeUniverse,
};

use crate::error_reporting::ErrorReporter;
use crate::program::Program;
use std::error::Error;

pub fn run_typechecker(
    program: &Program,
    _reporter: &mut ErrorReporter,
) -> Result<(Result<(TypeUniverse, SolvedTypes), usize>, usize), Box<dyn Error>> {
    Ok((Ok((TypeUniverse::new(), SolvedTypes::new(program))), 0))
}
