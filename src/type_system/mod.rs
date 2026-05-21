mod errors;
mod kinds;
mod local_inference;
mod operator_solver;
mod origin;
mod solving;

pub use errors::*;
pub use kinds::*;
pub use operator_solver::*;
pub use origin::*;
pub use solving::*;

use crate::error_reporting::ErrorReporter;
use crate::program::Program;
use std::error::Error;

#[allow(clippy::type_complexity)]
pub fn run_typechecker(
    program: &Program,
    reporter: &mut ErrorReporter,
) -> Result<(Result<(TypeUniverse, SolvedTypes), usize>, usize), Box<dyn Error>> {
    local_inference::run_typechecker_impl(program, reporter)
}
