mod errors;
mod kinds;
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

pub fn run_typechecker(
    program: &Program,
    _reporter: &mut ErrorReporter,
) -> Result<(Result<(TypeUniverse, SolvedTypes), usize>, usize), Box<dyn Error>> {
    Ok((Ok((TypeUniverse::new(), SolvedTypes::new(program))), 0))
}
