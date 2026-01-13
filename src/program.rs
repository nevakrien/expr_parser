use crate::macros::{Macro, expand_macros_recursive};
use crate::parsing::{Expr, LExpr, Loc, Located, Parser, Token};
use std::collections::HashMap;
use thiserror::Error;

pub type CResult<T> = Result<T, CompileError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CompileError {
    #[error("{s}")]
    SimpleError { loc: Loc, s: &'static str },

    #[error("{call_name} expected {expected} arguments, got {got}")]
    Arity {
        loc: Loc,
        call_name: &'static str,
        expected: usize,
        got: usize,
    },

    #[error(transparent)]
    Parse(#[from] crate::parsing::ParseError),
}

#[derive(Debug, Default)]
pub struct Program {
    pub macros: HashMap<String, Macro>,
    pub functions: Vec<LExpr>,
    pub structs: Vec<LExpr>,
    pub enums: Vec<LExpr>,
    pub unions: Vec<LExpr>,
}

impl Program {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_macro(&mut self, name: String, macro_def: Macro) {
        self.macros.insert(name, macro_def);
    }

    pub fn get_macro(&self, name: &str) -> Option<&Macro> {
        self.macros.get(name)
    }
}

pub struct ProgramParser<'a> {
    parser: Parser<'a>,
}

impl<'a> ProgramParser<'a> {
    pub fn new(src: &'a str, file: usize) -> Self {
        Self {
            parser: Parser::new(src, file),
        }
    }

    pub fn is_empty(&mut self) -> bool {
        self.parser.is_empty()
    }

    pub fn consume_expr(
        &mut self,
        program: &mut Program,
        on_expr: &mut dyn FnMut(&LExpr),
    ) -> CResult<bool> {
        let Some(mut expr) = self.parser.parse_stmt()? else {
            return Ok(false);
        };
        expand_macros_recursive(&mut expr, program)?;
        on_expr(&expr);

        self.handle_definition(expr, program)?;
        Ok(true)
    }

    fn handle_definition(&mut self, expr: LExpr, program: &mut Program) -> CResult<()> {
        let Located { loc: _, value } = expr;
        match value {
            Expr::Postfix(op, mut items) if op.value.as_str() == ";" => {
                self.handle_definition(items.pop().expect("bad structure"), program)
            }
            Expr::Prefix(open, items) if open.value.as_str() == "{" => {
                for item in items {
                    self.handle_definition(item, program)?;
                }
                Ok(())
            }
            Expr::Bin(eq, box_pair) if eq.value.as_str() == "=" => {
                let (lhs, rhs) = *box_pair;
                self.handle_assignment(lhs, rhs, program)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn handle_assignment(&mut self, lhs: LExpr, rhs: LExpr, program: &mut Program) -> CResult<()> {
        let Located {
            loc: rhs_loc,
            value: rhs_value,
        } = rhs;

        match rhs_value {
            Expr::Prefix(macro_kw, args) if macro_kw.value.as_str() == "macro" => {
                let name = get_single_ident(lhs)?;
                let macro_def = Macro::new(args, rhs_loc)?;
                program.add_macro(name, macro_def);
                Ok(())
            }
            Expr::Prefix(fn_kw, args)
                if fn_kw.value.as_str() == "fn" || fn_kw.value.as_str() == "cfn" =>
            {
                program.functions.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(fn_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(struct_kw, args) if struct_kw.value.as_str() == "struct" => {
                program.structs.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(struct_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(enum_kw, args) if enum_kw.value.as_str() == "enum" => {
                program.enums.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(enum_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(union_kw, args) if union_kw.value.as_str() == "union" => {
                program.unions.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(union_kw, args),
                });
                Ok(())
            }
            _ => Err(CompileError::SimpleError {
                loc: lhs.loc,
                s: "Unsupported definition",
            }),
        }
    }
}

pub fn get_single_ident(expr: LExpr) -> CResult<String> {
    match expr.value {
        Expr::Atom(Token::Ident(name)) => Ok(name),
        _ => Err(CompileError::SimpleError {
            loc: expr.loc,
            s: "Expected single identifier for macro name",
        }),
    }
}
