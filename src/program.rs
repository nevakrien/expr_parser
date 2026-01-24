use crate::error_messages::{
    ERR_EXPECTED_DEFINITION_VALUE, ERR_EXPECTED_SIMPLE_NAME, ERR_UNRESOLVED_NAME,
};
use crate::ir::LValue;
use crate::ir::NameId;
use crate::ir::TValue;
use crate::ir::TypeInfo;
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
pub enum Defined {
    ToBeDefined,
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
    pub pending_names: HashMap<NameId, Loc>,
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
            pending_names: HashMap::new(),
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

    /// Insert a new binding into the global scope, always creating a fresh ID.
    pub fn insert_value_in_global_scope(&mut self, name: String) -> NameId {
        let id = self.fresh_name_id();
        if let Some(value_scope) = self.scopes.first_mut() {
            value_scope.insert(name, id);
        } else {
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
    pub(crate) fn resolve_name(&mut self, loc: &Loc, name: String) -> CResult<NameId> {
        for value_scope in self.scopes.iter().rev() {
            if let Some(id) = value_scope.get(&name) {
                return Ok(*id);
            }
        }
        let id = self.insert_value_in_global_scope(name.to_string());
        self.definitions.insert(id, Defined::ToBeDefined);
        self.pending_names.entry(id).or_insert_with(|| loc.clone());
        Ok(id)
    }


    #[inline]
    pub fn with_scope<T>(&mut self, f: impl FnOnce(&mut Program) -> CResult<T>) -> CResult<T> {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    pub fn gather_definition(&mut self, expr: LExpr) -> CResult<()> {
        let Located { loc, value } = expr;
        match value {
            Expr::Postfix(op, mut items) if op.value == ";" => {
                self.gather_definition(items.pop().expect("bad structure"))
            }
            Expr::Prefix(open, items) if open.value == "{" => {
                for item in items {
                    self.gather_definition(item)?;
                }
                Ok(())
            }
            Expr::Bin(eq, box_pair) if eq.value == "=" => {
                let (lhs, rhs) = *box_pair;
                self.handle_assignment(lhs, rhs)
            }
            _ => {
                self.lower_value(Located { loc, value })?;
                Ok(())
            }
        }
    }

    pub fn compile_all(&mut self, parser: &mut Parser<'_>) -> CResult<()> {
        while !parser.is_empty() {
            match parser.parse_with_macros(self)? {
                Some(expr) => self.gather_definition(expr)?,
                None => break,
            }
        }

        self.check_pending_names()
    }

    pub fn check_pending_names(&self) -> CResult<()> {
        if self.pending_names.is_empty() {
            return Ok(());
        }

        if let Some((_, loc)) = self.pending_names.iter().next() {
            return Err(CompileError::SimpleError {
                loc: loc.clone(),
                s: ERR_UNRESOLVED_NAME,
            });
        }

        Ok(())
    }

    fn handle_assignment(&mut self, lhs: LExpr, rhs: LExpr) -> CResult<()> {
        let Located {
            loc: rhs_loc,
            value: rhs_value,
        } = rhs;

        let name = match lhs.value {
            Expr::Atom(Token::Ident(name)) => {
                if let Some(id) = self
                    .scopes
                    .last()
                    .and_then(|scope| scope.get(&name))
                    .copied()
                {
                    if matches!(self.definitions.get(&id), Some(Defined::ToBeDefined)) {
                        id
                    } else {
                        self.insert_value_in_current_scope(name)
                    }
                } else {
                    self.insert_value_in_current_scope(name)
                }
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: lhs.loc,
                    s: ERR_EXPECTED_SIMPLE_NAME,
                });
            }
        };

        self.pending_names.remove(&name);

        let def: Defined = match rhs_value {
            Expr::Prefix(macro_kw, args) if macro_kw.value == "macro" => {
                let macro_def = Macro::new(args, rhs_loc)?;
                Defined::Macro(macro_def)
            }
            Expr::Prefix(ref fn_kw, _) if fn_kw.value == "fn" || fn_kw.value == "cfn" => self
                .with_scope(|prog| {
                    let v = prog.lower_value(Located {
                        loc: rhs_loc,
                        value: rhs_value,
                    })?;
                    Ok(Defined::Value(v))
                })?,

            Expr::Prefix(ref fn_kw, _)
                if fn_kw.value == "struct" || fn_kw.value == "enum" || fn_kw.value == "union" =>
            {
                Defined::Raw(Located {
                    loc: rhs_loc,
                    value: rhs_value,
                })
            }
            _ => {
                return Err(CompileError::SimpleError {
                    loc: rhs_loc,
                    s: ERR_EXPECTED_DEFINITION_VALUE,
                });
            }
        };

        self.definitions.insert(name, def);

        Ok(())
    }
}
