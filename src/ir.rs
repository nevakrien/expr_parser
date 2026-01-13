use crate::parsing::{Expr, FixedToken, LExpr, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};
use std::collections::HashMap;

// Type aliases
pub type LName = Located<NameId>;
pub type TName = Typed<NameId>;
pub type TDecl = Typed<Decl>;
pub type TValue = Typed<Value>;
pub type TPattern = Typed<Pattern>;

// Core type definitions
#[derive(Debug, Clone, PartialEq)]
pub struct TypeInfo {
    pub uses: Vec<Loc>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Typed<T> {
    pub loc: Loc,
    pub ty: Option<TypeInfo>,
    pub value: T,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub u32);

#[derive(Debug, Clone, PartialEq)]
pub struct ProgramIr {
    pub decls: Vec<TDecl>,
    pub body: TValue,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Generics {
    pub params: Vec<TPattern>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    RuntimeValue {
        name: LName,
        generics: Option<Generics>,
        value: TValue,
    },
    Struct {
        name: LName,
        generics: Option<Generics>,
        fields: Vec<(LName, TPattern)>,
    },
    Union {
        name: LName,
        generics: Option<Generics>,
        fields: Vec<(LName, TPattern)>,
    },
    Enum {
        name: LName,
        generics: Option<Generics>,
        variants: Vec<EnumVariant>,
    },
    Alias {
        name: LName,
        generics: Option<Generics>,
        ty: TPattern,
    },
    Macro {
        name: LName,
        generics: Option<Generics>,
        params: Vec<LName>,
        body: TValue,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: LName,
    pub fields: Vec<TPattern>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    NameRef(NameId),
    Literal(Literal),
    Call {
        callee: Box<TValue>,
        args: Vec<TValue>,
    },
    Index {
        base: Box<TValue>,
        args: Vec<TValue>,
    },
    Block {
        statements: Vec<TValue>,
        return_value: Option<Box<TValue>>,
    },
    Let {
        pat: TPattern,
        value: Box<TValue>,
    },
    Assign {
        target: Box<TValue>,
        value: Box<TValue>,
    },
    If {
        cond: Box<TValue>,
        then: Box<TValue>,
        els: Option<Box<TValue>>,
    },
    While {
        cond: Box<TValue>,
        body: Box<TValue>,
    },
    Func {
        params: Vec<Param>,
        ret: Option<TPattern>,
        body: Box<TValue>,
    },
    Return(Option<Box<TValue>>),
    Break,
    Continue,
    Cast {
        value: Box<TValue>,
        ty: TPattern,
    },
    TypeAnnotation {
        value: Box<TValue>,
        ty: TPattern,
    },
    Match {
        value: Box<TValue>,
        arms: Vec<MatchArm>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Str(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub pat: TPattern,
    pub ty: Option<TPattern>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Bind(NameId),
    Wildcard,
    Tuple(Vec<TPattern>),
    Destructure {
        ctor: LName,
        fields: Vec<PatternField>,
    },
    GenericSpecialization {
        base: Box<TPattern>,
        args: Vec<TPattern>,
    },
    Literal(Literal),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PatternField {
    pub name: LName,
    pub value: TPattern,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pat: TPattern,
    pub guard: Option<TValue>,
    pub body: TValue,
}

// Implementations
impl<T> Typed<T> {
    pub fn with<U>(&self, value: U) -> Typed<U> {
        Typed {
            loc: self.loc.clone(),
            ty: self.ty.clone(),
            value,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Typed<U> {
        Typed {
            loc: self.loc,
            ty: self.ty,
            value: f(self.value),
        }
    }

    pub fn into_located(self) -> Located<T> {
        Located {
            loc: self.loc,
            value: self.value,
        }
    }
}

impl<T> From<Located<T>> for Typed<T> {
    fn from(value: Located<T>) -> Self {
        Typed {
            loc: value.loc,
            ty: None,
            value: value.value,
        }
    }
}

impl ProgramIr {
    pub fn empty(loc: Loc) -> Self {
        Self {
            decls: Vec::new(),
            body: Typed {
                loc,
                ty: None,
                value: Value::Block {
                    statements: Vec::new(),
                    return_value: None,
                },
            },
        }
    }
}

// IR Lowerer
#[derive(Debug)]
pub struct IrLowerer<'a> {
    program: &'a Program,
    next_name_id: u32,
    scopes: Vec<(HashMap<String, NameId>, HashMap<String, NameId>)>,
}

impl<'a> IrLowerer<'a> {
    pub fn new(program: &'a Program) -> Self {
        Self {
            program,
            next_name_id: 0,
            scopes: Vec::new(),
        }
    }

    fn lower_value(&mut self, expr: LExpr) -> CResult<TValue> {
    	/*
    	TODOS: this function is very much half baked
    	1. we need to move macro expantion into here so macros respect scope
    	2. index and othe similar ops dont respect the fact they might be parsing patterns
    	*/
        match expr.value {
            Expr::Atom(token) => self.lower_atom(&expr.loc, token),
            Expr::Prefix(open, mut items) if open.value == FixedToken::LBrace => {
                self.push_scope();
                let mut statements = Vec::new();
                let mut return_value = None;

                if let Some(last) = items.pop() {
                    for item in items {
                        statements.push(self.lower_value(item)?);
                    }

                    if matches!(last.value, Expr::Atom(Token::Operator(FixedToken::Semi))) {
                        return_value = None;
                    } else {
                        return_value = Some(Box::new(self.lower_value(last)?));
                    }
                }

                self.pop_scope();
                Ok(self.typed_value(
                    expr.loc,
                    Value::Block {
                        statements,
                        return_value,
                    },
                ))
            }
            Expr::Prefix(open, items) if open.value == FixedToken::Let => {
                let loc = expr.loc.clone();
                let (pat_expr, value_expr) = split_prefix(loc.clone(), "let", items)?;
                let pat = self.lower_pattern(pat_expr)?;
                let value = Box::new(self.lower_value(value_expr)?);
                Ok(self.typed_value(loc, Value::Let { pat, value }))
            }
            Expr::Postfix(open, items) if open.value == FixedToken::LParen => {
                let loc = expr.loc.clone();
                let (callee_expr, args_exprs) = split_postfix(loc.clone(), "call", items)?;
                let callee = Box::new(self.lower_value(callee_expr)?);
                let mut args = Vec::new();
                for arg in args_exprs {
                    args.push(self.lower_value(arg)?);
                }
                Ok(self.typed_value(loc, Value::Call { callee, args }))
            }
            Expr::Postfix(open, items) if open.value == FixedToken::LBracket => {
                let loc = expr.loc.clone();
                //this is actually wrong as we may parse a generic specilization
                let (base_expr, args_exprs) = split_postfix(loc.clone(), "index", items)?;
                let base = Box::new(self.lower_value(base_expr)?);
                let mut args = Vec::new();
                for arg in args_exprs {
                    args.push(self.lower_value(arg)?);
                }
                Ok(self.typed_value(loc, Value::Index { base, args }))
            }
            Expr::Bin(op, pair) if op.value == FixedToken::Assign => {
                let (lhs, rhs) = pair.as_ref();
                let target = Box::new(self.lower_value(lhs.clone())?);
                let value = Box::new(self.lower_value(rhs.clone())?);
                Ok(self.typed_value(expr.loc, Value::Assign { target, value }))
            }
            Expr::Prefix(open, items) if open.value == FixedToken::Match => {
                let loc = expr.loc.clone();
                let (value_expr, arms_expr) = split_prefix(loc.clone(), "match", items)?;
                let value = Box::new(self.lower_value(value_expr)?);
                let arms = self.lower_match_arm(arms_expr)?;
                Ok(self.typed_value(loc, Value::Match { value, arms }))
            }
            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Unsupported expression in IR lowering",
            }),
        }
    }

    fn lower_atom(&mut self, loc: &Loc, token: Token) -> CResult<TValue> {
        let value = match token {
            Token::NumLit(n) => Value::Literal(Literal::Num(n)),
            Token::FloatLit(f) => Value::Literal(Literal::Float(f)),
            Token::StrLit(s) => Value::Literal(Literal::Str(s)),
            Token::Ident(name) => {
                let id = self.resolve_or_insert_value(&name);
                Value::NameRef(id)
            }
            Token::Operator(_) => {
                return Err(CompileError::SimpleError {
                    loc: loc.clone(),
                    s: "Unexpected operator atom in IR lowering",
                });
            }
        };
        Ok(self.typed_value(loc.clone(), value))
    }

    fn lower_pattern(&mut self, expr: LExpr) -> CResult<TPattern> {
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => {
                Ok(self.typed_pattern(expr.loc, Pattern::Wildcard))
            }
            Expr::Atom(Token::Ident(name)) => {
                let id = self.resolve_or_insert_value(&name);
                Ok(self.typed_pattern(expr.loc, Pattern::Bind(id)))
            }
            Expr::Prefix(open, items) if open.value == FixedToken::LParen => {
                let mut parts = Vec::new();
                for item in items {
                    parts.push(self.lower_pattern(item)?);
                }
                Ok(self.typed_pattern(expr.loc, Pattern::Tuple(parts)))
            }
            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Unsupported pattern in IR lowering",
            }),
        }
    }

    fn typed_value(&self, loc: Loc, value: Value) -> TValue {
        Typed {
            loc,
            ty: None,
            value,
        }
    }

    fn typed_pattern(&self, loc: Loc, value: Pattern) -> TPattern {
        Typed {
            loc,
            ty: None,
            value,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push((HashMap::new(), HashMap::new()));
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn resolve_or_insert_value(&mut self, name: &str) -> NameId {
        for (value_scope, _) in self.scopes.iter().rev() {
            if let Some(id) = value_scope.get(name) {
                return *id;
            }
        }
        let id = self.fresh_name_id();
        if let Some((value_scope, _)) = self.scopes.last_mut() {
            value_scope.insert(name.to_string(), id);
        }
        id
    }

    fn fresh_name_id(&mut self) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        id
    }

    fn lower_match_arm(&mut self, expr: LExpr) -> CResult<Vec<MatchArm>> {
        match expr.value {
            Expr::Prefix(open, items) if open.value == FixedToken::LBrace => {
                let mut arms = Vec::new();
                for arm_expr in items {
                    arms.push(self.lower_single_match_arm(arm_expr)?);
                }
                Ok(arms)
            }
            _ => {
                // Single arm without braces
                Ok(vec![self.lower_single_match_arm(expr)?])
            }
        }
    }

    fn lower_single_match_arm(&mut self, expr: LExpr) -> CResult<MatchArm> {
        match expr.value {
            Expr::Prefix(open, items) if open.value == FixedToken::FatArrow => {
                let (pat_expr, body_expr) = split_prefix(expr.loc, "=>", items)?;
                let pat = self.lower_pattern(pat_expr)?;
                let body = self.lower_value(body_expr)?;
                Ok(MatchArm {
                    pat,
                    guard: None,
                    body,
                })
            }
            Expr::Bin(op, pair) if op.value == FixedToken::If => {
                let (left, right) = pair.as_ref();
                // This is a guard: pattern if guard => body
                if let Expr::Prefix(open, items) = &left.value {
                    if open.value == FixedToken::FatArrow && items.len() == 2 {
                        let pat_expr = items[0].clone();
                        let body_expr = items[1].clone();
                        let pat = self.lower_pattern(pat_expr)?;
                        let guard = self.lower_value(right.clone())?;
                        let body = self.lower_value(body_expr)?;
                        return Ok(MatchArm {
                            pat,
                            guard: Some(guard),
                            body,
                        });
                    }
                }
                Err(CompileError::SimpleError {
                    loc: expr.loc,
                    s: "Invalid match arm guard syntax",
                })
            }
            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Invalid match arm syntax",
            }),
        }
    }
}

// Public API functions
pub fn lower_expr(program: &Program, expr: LExpr) -> CResult<TValue> {
    IrLowerer::new(program).lower_value(expr)
}

// Helper functions
fn split_prefix(loc: Loc, name: &'static str, items: Vec<LExpr>) -> CResult<(LExpr, LExpr)> {
    if items.len() != 2 {
        return Err(CompileError::Arity {
            loc,
            call_name: name,
            expected: 2,
            got: items.len(),
        });
    }
    let mut iter = items.into_iter();
    Ok((iter.next().unwrap(), iter.next().unwrap()))
}

fn split_postfix(
    loc: Loc,
    name: &'static str,
    mut items: Vec<LExpr>,
) -> CResult<(LExpr, Vec<LExpr>)> {
    if let Some(first) = items.pop() {
        Ok((first, items))
    } else {
        Err(CompileError::Arity {
            loc,
            call_name: name,
            expected: 1,
            got: 0,
        })
    }
}

fn fallback_loc(exprs: &[LExpr]) -> Loc {
    exprs.first().map(|expr| expr.loc.clone()).unwrap_or(Loc {
        range: 0..0,
        file: 0,
    })
}
