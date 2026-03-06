pub mod error_reporting;
pub mod global_type_inference;
pub mod identity_hasher;
pub mod ir;
pub mod lifetime_graph;
#[cfg(feature = "solver_order_fuzz")]
mod local_solver_order;
pub mod local_type_inference;
pub mod macros;
pub mod parsing;
pub mod program;
pub mod string_intern;
pub mod struct_layout;
pub mod type_inference;

pub use error_reporting::ErrorReporter;
pub use parsing::{Expr, LExpr, Parser, Token};
pub use program::{CompileError, Program};

#[cfg(all(feature = "solver_order_fuzz", feature = "determinism"))]
compile_error!("features `solver_order_fuzz` and `determinism` are incompatible; use only one");
