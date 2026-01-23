use crate::error_messages::{ERR_EXPECTED_DEFINITION_VALUE, ERR_EXPECTED_SIMPLE_NAME};
use crate::ir::LValue;
use crate::ir::NameId;
use crate::ir::TValue;
use crate::ir::TypeInfo;
use crate::macros::{expand_macros_recursive, Macro};
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
pub enum Defined {
    Raw(LExpr),
    Value(TValue),
    Type(LValue),
    Macro(Macro),
}

#[derive(Debug)]
pub struct Program {
    pub definitions: HashMap<NameId, Defined>,
    pub current_infrence: Vec<TypeInfo>,

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
            definitions: HashMap::new(),
            current_infrence: Vec::new(),

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

    pub fn get_macro(&self, name: &str) -> Option<&Macro> {
        //TODO think if we wana do scopes
        let id = self.scopes[0].get(name)?;
        if let Some(Defined::Macro(ans)) = self.definitions.get(id) {
            Some(ans)
        } else {
            None
        }
    }
}

impl<'a> Parser<'a> {
    pub fn parse_with_macros(&mut self, program: &Program) -> CResult<Option<LExpr>> {
        let Some(mut expr) = self.parse_stmt()? else {
            return Ok(None);
        };
        expand_macros_recursive(&mut expr, program)?;
        Ok(Some(expr))
    }
}

impl Program {
    #[inline]
    pub fn with_scope<T>(&mut self, f: impl FnOnce(&mut Program) -> CResult<T>) -> CResult<T> {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    pub fn handle_definition(&mut self, expr: LExpr) -> CResult<()> {
        let Located { loc, value } = expr;
        match value {
            Expr::Postfix(op, mut items) if op.value == ";" => {
                self.handle_definition(items.pop().expect("bad structure"))
            }
            Expr::Prefix(open, items) if open.value == "{" => {
                for item in items {
                    self.handle_definition(item)?;
                }
                Ok(())
            }
            Expr::Bin(eq, box_pair) if eq.value == "=" => {
                let (lhs, rhs) = *box_pair;
                self.handle_assignment(lhs, rhs)?;
                Ok(())
            }
            _ => Ok({self.lower_value(Located { loc, value })?;}),
        }
    }

    fn handle_assignment(&mut self, lhs: LExpr, rhs: LExpr) -> CResult<()> {
        let Located {
            loc: rhs_loc,
            value: rhs_value,
        } = rhs;

        let name = match lhs.value {
            Expr::Atom(Token::Ident(name)) => self.insert_value_in_current_scope(name),
            _ => {
                return Err(CompileError::SimpleError {
                    loc: lhs.loc,
                    s: ERR_EXPECTED_SIMPLE_NAME,
                });
            }
        };

        let def: Defined = self.with_scope(|prog|Ok(match rhs_value {
            Expr::Prefix(macro_kw, args) if macro_kw.value == "macro" => {
                let macro_def = Macro::new(args, rhs_loc)?;
                // self.add_macro(name, macro_def);
                Defined::Macro(macro_def)
            }
            Expr::Prefix(ref fn_kw, ref _args) if fn_kw.value == "fn" || fn_kw.value == "cfn" => {
                let v = prog.lower_value(Located {
                    loc: rhs_loc,
                    value: rhs_value,
                })?;
                Defined::Value(v)
            }
            Expr::Prefix(struct_kw, args) if struct_kw.value == "struct" => {
                let r = Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(struct_kw, args),
                };

                Defined::Raw(r)
            }
            Expr::Prefix(enum_kw, args) if enum_kw.value == "enum" => {
                let r = Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(enum_kw, args),
                };

                Defined::Raw(r)
            }
            Expr::Prefix(union_kw, args) if union_kw.value == "union" => {
                let r = Located {
                    loc: rhs_loc,
                    value: Expr::Prefix(union_kw, args),
                };
                Defined::Raw(r)
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: rhs_loc,
                    s: ERR_EXPECTED_DEFINITION_VALUE,
                });
            }
        }))?;

        self.definitions.insert(name, def);

        Ok(())
    }
}
