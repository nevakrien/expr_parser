use crate::error_messages::{ERR_EXPECTED_MACRO_NAME, ERR_UNSUPPORTED_DEFINITION};
use crate::ir::NameId;
use crate::ir::TValue;
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

    #[error("{message}")]
    UnsupportedForm {
        loc: Loc,
        op_loc: Option<Loc>,
        op: Option<&'static str>,
        message: &'static str,
    },

    #[error(transparent)]
    Parse(#[from] crate::parsing::ParseError),
}

#[derive(Debug)]
pub struct Program {
    pub macros: HashMap<String, Macro>,
    pub functions: Vec<TValue>,
    pub structs: Vec<LExpr>,
    pub enums: Vec<LExpr>,
    pub unions: Vec<LExpr>,

    pub next_name_id: usize,
    pub scopes: Vec<HashMap<String, NameId>>,
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    pub fn new() -> Self {
        let mut program = Self {
            macros: HashMap::new(),
            functions: Vec::new(),
            structs: Vec::new(),
            enums: Vec::new(),
            unions: Vec::new(),
            next_name_id: 0,
            scopes: vec![HashMap::new()],
        };
        program.insert_builtin_types();
        program
    }

    fn insert_builtin_types(&mut self) {
        for name in ["int", "float", "bool", "str", "void"] {
            self.insert_value_in_current_scope(name.to_string());
        }
    }

    /// Push a new variable scope onto the stack
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the current variable scope
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Insert a new binding into the current (innermost) scope, always creating a fresh ID.
    pub fn insert_value_in_current_scope(&mut self, name: String) -> NameId {
        let id = self.fresh_name_id();
        if let Some(value_scope) = self.scopes.last_mut() {
            value_scope.insert(name, id);
        } else {
            // If you ever lower without at least one scope pushed, that's a bug.
            debug_assert!(false, "no scope available when inserting binding");
        }
        id
    }

    /// Generate a fresh unique name ID
    fn fresh_name_id(&mut self) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        id
    }

    pub fn add_macro(&mut self, name: String, macro_def: Macro) {
        self.macros.insert(name, macro_def);
    }

    pub fn get_macro(&self, name: &str) -> Option<&Macro> {
        self.macros.get(name)
    }
}

impl<'a> Parser<'a> {
    pub fn compile_expr(
        &mut self,
        program: &mut Program,
        on_expr: &mut dyn FnMut(&LExpr),
    ) -> CResult<bool> {
        let Some(mut expr) = self.parse_stmt()? else {
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
            Expr::Postfix(op, mut items) if op.value == ";" => {
                self.handle_definition(items.pop().expect("bad structure"), program)
            }
            Expr::Prefix(open, items) if open.value == "{" => {
                for item in items {
                    self.handle_definition(item, program)?;
                }
                Ok(())
            }
            Expr::Bin(eq, box_pair) if eq.value == "=" => {
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
            Expr::Prefix(macro_kw, args) if macro_kw.value == "macro" => {
                let name = get_single_ident(lhs)?;
                let macro_def = Macro::new(args, rhs_loc)?;
                program.add_macro(name, macro_def);
                Ok(())
            }
            Expr::Prefix(ref fn_kw, ref _args) if fn_kw.value == "fn" || fn_kw.value == "cfn" => {
                let v = program.lower_value(Located {
                    loc: rhs_loc,
                    value: rhs_value,
                })?;
                program.functions.push(v);
                Ok(())
            }
            Expr::Prefix(struct_kw, args) if struct_kw.value == "struct" => {
                program.structs.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(struct_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(enum_kw, args) if enum_kw.value == "enum" => {
                program.enums.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(enum_kw, args),
                });
                Ok(())
            }
            Expr::Prefix(union_kw, args) if union_kw.value == "union" => {
                program.unions.push(Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(union_kw, args),
                });
                Ok(())
            }
            _ => Err(CompileError::SimpleError {
                loc: lhs.loc,
                s: ERR_UNSUPPORTED_DEFINITION,
            }),
        }
    }
}

pub fn get_single_ident(expr: LExpr) -> CResult<String> {
    match expr.value {
        Expr::Atom(Token::Ident(name)) => Ok(name),
        _ => Err(CompileError::SimpleError {
            loc: expr.loc,
            s: ERR_EXPECTED_MACRO_NAME,
        }),
    }
}
