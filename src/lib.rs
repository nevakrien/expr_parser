pub mod data_structures;
pub mod error_reporting;
pub mod ir;
pub mod macros;
pub mod parsing;
pub mod program;
pub mod requirment;
pub mod type_system;

//this we fixup later
// pub mod struct_layout;

pub use error_reporting::ErrorReporter;
pub use parsing::{Expr, LExpr, Parser, Token};
pub use program::{CompileError, Program};
