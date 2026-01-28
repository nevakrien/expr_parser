use crate::error_messages::{ERR_EXPECTED_DEFINITION_VALUE, ERR_EXPECTED_SIMPLE_NAME};
use crate::ir::IValue;
use crate::ir::NameId;
use crate::macros::{expand_macros_recursive, Macro};
use crate::parsing::{Expr, LExpr, Loc, Located, Parser, Token};
use crate::string_intern::StrId;
use crate::string_intern::StringInterner;
use crate::type_inference::TypeId;
use crate::type_inference::TypeValue;
use std::collections::HashMap;
use thiserror::Error;

pub type CResult<T> = Result<T, CompileError>;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum CompileError {
    #[error("{s}")]
    SimpleError { loc: Loc, s: &'static str },

    #[error("Unresolved name")]
    UnresolvedNames { name: String, locs: Vec<Loc> },

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
    Value(IValue),
    Type { val: IValue, ty: TypeId },
    // TypeRef(TypeId),
    BuildinType(TypeValue),
    Macro(Macro),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValId(pub usize);

#[derive(Debug, Clone, PartialEq)]
pub struct WithId<T> {
    pub value: T,
    pub id: ValId,
}

impl<T> WithId<T> {
    /// Create a new Typed value with the same location and type but different inner value
    pub fn with<U>(&self, value: U) -> WithId<U> {
        WithId { id: self.id, value }
    }

    /// Transform the inner value using a function while preserving location and type
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> WithId<U> {
        WithId {
            id: self.id,
            value: f(self.value),
        }
    }
}

#[derive(Debug)]
pub struct Program {
    pub definitions: HashMap<NameId, Defined>,
    // pub current_infrence: Vec<TypeInfo>,
    // pub type_store: TypeStore,
    val_locs: Vec<Loc>,

    names_strs: Vec<StrId>,
    pub str_intern: StringInterner,

    pub scopes: Vec<HashMap<StrId, NameId>>,
    pub pending_names: HashMap<NameId, Vec<Loc>>,
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
            // type_store: TypeStore::new(),
            val_locs: Vec::new(),

            names_strs: Vec::new(),
            str_intern: StringInterner::new(),

            scopes: vec![HashMap::new()],
            pending_names: HashMap::new(),
        };
        program.insert_builtin_types();
        program
    }

    pub fn with_id<T>(&mut self, x: Located<T>) -> WithId<T> {
        let id = ValId(self.val_locs.len());
        self.val_locs.push(x.loc);
        WithId { value: x.value, id }
    }

    pub fn id_value<T>(&mut self, loc: Loc, value: T) -> WithId<T> {
        self.with_id(loc.clone().with(value))
    }

    pub fn get_loc(&self, v: ValId) -> Loc {
        self.val_locs[v.0].clone()
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
    pub fn insert_value_in_current_scope(&mut self, name: StrId) -> NameId {
        let id = self.fresh_name_id(name);
        let value_scope = self
            .scopes
            .last_mut()
            .expect("no scope available when inserting binding");
        value_scope.insert(name, id);
        id
    }

    /// Insert a new binding into the global scope, always creating a fresh ID.
    pub fn insert_value_in_global_scope(&mut self, name: StrId) -> NameId {
        let id = self.fresh_name_id(name);
        if let Some(value_scope) = self.scopes.first_mut() {
            value_scope.insert(name, id);
        } else {
            debug_assert!(false, "no scope available when inserting binding");
        }
        id
    }

    /// Generate a fresh unique name ID
    fn fresh_name_id(&mut self, s: StrId) -> NameId {
        let id = NameId(self.names_strs.len());
        self.names_strs.push(s);
        id
    }

    pub fn get_macro(&mut self, name: &str) -> Option<&Macro> {
        //TODO think if we wana do scopes
        let name = self.str_intern.intern(name);
        let id = self.scopes[0].get(&name)?;
        if let Some(Defined::Macro(ans)) = self.definitions.get(id) {
            Some(ans)
        } else {
            None
        }
    }
}

impl<'a> Parser<'a> {
    pub fn parse_with_macros(&mut self, program: &mut Program) -> CResult<Option<LExpr>> {
        let Some(mut expr) = self.parse_stmt()? else {
            return Ok(None);
        };
        expand_macros_recursive(&mut expr, program)?;
        Ok(Some(expr))
    }
}

impl Program {
    //this takes a String because a name is mentioned once as String and gets resolved here
    //it actually has a much nicer cache behvior becaused the String usually gets freed and is giving something else room
    //instead of a &str which would bring cache line to some random parse data
    pub(crate) fn resolve_name(&mut self, loc: &Loc, name: &str) -> CResult<NameId> {
        let name = self.str_intern.intern(name);
        for value_scope in self.scopes.iter().skip(1).rev() {
            if let Some(id) = value_scope.get(&name) {
                return Ok(*id);
            }
        }

        if let Some(id) = self.scopes[0].get(&name) {
            //might still be empty.
            if let Some(spot) = self.pending_names.get_mut(id) {
                spot.push(loc.clone())
            }
            return Ok(*id);
        }

        //errors get reported later it can be there is just a late mention so we dont know yet
        let id = self.insert_value_in_global_scope(name);
        self.definitions.insert(id, Defined::ToBeDefined);
        self.pending_names.entry(id).or_default().push(loc.clone());
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

    pub fn lower_all(&mut self, parser: &mut Parser<'_>) -> CResult<()> {
        while !parser.is_empty() {
            match parser.parse_with_macros(self)? {
                //TODO when doing multi Error this ? should be an if let Err
                //     we can put the result here to a multi error since gathering is independent
                Some(expr) => self.gather_definition(expr)?,
                None => break,
            }
        }

        self.check_pending_names()
    }

    pub fn check_pending_names(&mut self) -> CResult<()> {
        if self.pending_names.is_empty() {
            return Ok(());
        }

        for (id, locs_ref) in self.pending_names.iter_mut() {
            match self.definitions.entry(*id) {
                std::collections::hash_map::Entry::Occupied(o) => {
                    if !matches!(o.get(), Defined::ToBeDefined) {
                        continue;
                    }

                    // o.remove().0
                    // o.get().clone() //needed for the repl
                }
                _ => continue,
            };
            let name = self.str_intern.resolve(self.names_strs[id.0]).to_string();

            //take locs out so we dont double report them
            let mut locs = Vec::new();
            std::mem::swap(locs_ref, &mut locs);

            //TODO return a multi error here when we add them
            return Err(CompileError::UnresolvedNames { name, locs });
        }
        Ok(()) //should never get here
    }

    fn handle_assignment(&mut self, lhs: LExpr, rhs: LExpr) -> CResult<()> {
        let Located {
            loc: rhs_loc,
            value: rhs_value,
        } = rhs;

        let (_, name) = match lhs.value {
            Expr::Atom(Token::Ident(name)) => {
                let name = self.str_intern.intern(&name);
                if let Some(id) = self
                    .scopes
                    .last()
                    .and_then(|scope| scope.get(&name))
                    .copied()
                {
                    if matches!(self.definitions.get(&id), Some(Defined::ToBeDefined)) {
                        (name, id)
                    } else {
                        (name, self.insert_value_in_current_scope(name))
                    }
                } else {
                    (name, self.insert_value_in_current_scope(name))
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
