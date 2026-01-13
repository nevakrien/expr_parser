use crate::parsing::{Expr, FixedToken, LExpr, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};
use std::collections::HashMap;

// Type aliases for commonly used typed/located constructs
pub type LName = Located<NameId>; // Located name identifier
pub type TName = Typed<NameId>; // Typed name identifier
pub type TDecl = Typed<Decl>; // Typed declaration
pub type TValue = Typed<Value>; // Typed value/expression
pub type TPattern = Typed<Pattern>; // Typed pattern

// Core type definitions for the IR
#[derive(Debug, Clone, PartialEq)]
pub struct TypeInfo {
    /// Locations where this type is used in the code
    pub uses: Vec<Loc>,
}

/// Wrapper that adds location and type information to any value
#[derive(Debug, Clone, PartialEq)]
pub struct Typed<T> {
    /// Source location of this construct
    pub loc: Loc,
    /// Type information (if available/known)
    pub ty: Option<TypeInfo>,
    /// The underlying value
    pub value: T,
}

/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub u32);

/// The intermediate representation (IR) of a complete program
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramIr {
    /// Global/top-level declarations
    pub decls: Vec<TDecl>,
    /// Main program body expression
    pub body: TValue,
}

/// Generic parameters for declarations
#[derive(Debug, Clone, PartialEq)]
pub struct Generics {
    /// Parameter patterns (typically type variables)
    pub params: Vec<TPattern>,
}

/// Declaration of a type, constant, or macro in the global scope
///
/// TODO: Eventually support local declarations within functions
/// TODO: Macros should be handled here (explicitly or conceptually)
#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    /// Runtime value declaration (constants, functions, etc.)
    RuntimeValue {
        name: LName,
        generics: Option<Generics>,
        value: TValue,
    },
    /// Struct definition with named fields
    Struct {
        name: LName,
        generics: Option<Generics>,
        fields: Vec<(LName, TPattern)>,
    },
    /// Union definition with named fields (one field active at a time)
    Union {
        name: LName,
        generics: Option<Generics>,
        fields: Vec<(LName, TPattern)>,
    },
    /// Enum definition with variants
    Enum {
        name: LName,
        generics: Option<Generics>,
        variants: Vec<EnumVariant>,
    },
    /// Type alias declaration
    Alias {
        name: LName,
        generics: Option<Generics>,
        ty: TPattern,
    },
    /// Macro definition
    Macro {
        name: LName,
        generics: Option<Generics>,
        params: Vec<LName>,
        body: TValue,
    },
}

/// Single variant within an enum declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: LName,
    pub fields: Vec<TPattern>,
}

/// Runtime values including functions, closures, and control flow constructs
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Reference to a named variable or function
    NameRef(NameId),
    /// Literal value (number, string, etc.)
    Literal(Literal),
    /// Function/method call
    Call {
        callee: Box<TValue>,
        args: Vec<TValue>,
    },
    /// Indexing operation (may also represent generic specialization)
    Index {
        base: Box<TValue>,
        args: Vec<TValue>,
    },
    /// Block of statements with optional return value
    Block {
        statements: Vec<TValue>,
        return_value: Option<Box<TValue>>,
    },
    /// Variable binding declaration
    Let { pat: TPattern, value: Box<TValue> },
    /// Assignment operation
    Assign {
        target: Box<TValue>,
        value: Box<TValue>,
    },
    /// Conditional expression
    If {
        cond: Box<TValue>,
        then: Box<TValue>,
        els: Option<Box<TValue>>,
    },
    /// While loop
    While {
        cond: Box<TValue>,
        body: Box<TValue>,
    },
    /// Function literal/anonymous function
    Func {
        params: Vec<Param>,
        ret: Option<TPattern>,
        body: Box<TValue>,
    },
    /// Return from function (with optional value)
    Return(Option<Box<TValue>>),
    /// Break from loop
    Break,
    /// Continue to next loop iteration
    Continue,
    /// Explicit type cast
    Cast { value: Box<TValue>, ty: TPattern },
    /// Type annotation for expressions
    TypeAnnotation { value: Box<TValue>, ty: TPattern },
    /// Pattern matching expression
    Match {
        value: Box<TValue>,
        arms: Vec<MatchArm>,
    },
}

/// Literal values that can appear in the code
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Str(String),
}

/// Function parameter declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub pat: TPattern,
    pub ty: Option<TPattern>,
}

/// Patterns used for:
/// - Pattern matching (match expressions, function parameters)
/// - Type annotations (e.g., Option[Result[T, E]])
/// - Assignment targets (e.g., id[T] = fn(x: T) { x })
///
/// TODO: figure out if this should have a field for Value
///        this would come up for *x[T] = 2; or similar
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// Bind a value to a name (variable binding)
    Bind(NameId),
    /// Wildcard pattern that matches anything (_)
    Wildcard,
    /// Tuple pattern with multiple sub-patterns
    Tuple(Vec<TPattern>),
    /// Struct/enum destructoring pattern
    Destructure {
        ctor: LName,
        fields: Vec<PatternField>,
    },
    /// Generic type specialization (e.g., Foo[T, U])
    GenericSpecialization {
        base: Box<TPattern>,
        args: Vec<TPattern>,
    },
    /// Literal value pattern
    Literal(Literal),
}

/// Named field in a destructoring pattern
#[derive(Debug, Clone, PartialEq)]
pub struct PatternField {
    pub name: LName,
    pub value: TPattern,
}

/// Single arm in a match expression
#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pat: TPattern,
    pub guard: Option<TValue>,
    pub body: TValue,
}

// Implementations for IR types

impl<T> Typed<T> {
    /// Create a new Typed value with the same location and type but different inner value
    pub fn with<U>(&self, value: U) -> Typed<U> {
        Typed {
            loc: self.loc.clone(),
            ty: self.ty.clone(),
            value,
        }
    }

    /// Transform the inner value using a function while preserving location and type
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Typed<U> {
        Typed {
            loc: self.loc,
            ty: self.ty,
            value: f(self.value),
        }
    }

    /// Convert to a Located value (dropping type information)
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
    /// Create an empty program IR with a block body
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

/// Lowers parsed AST to IR with name resolution
///
/// Handles variable scoping, pattern matching, and expression transformation.
/// Names are resolved to unique IDs throughout the lowering process.
#[derive(Debug)]
pub struct IrLowerer<'a> {
    program: &'a Program,
    next_name_id: u32,
    scopes: Vec<(HashMap<String, NameId>, HashMap<String, NameId>)>,
}

impl<'a> IrLowerer<'a> {
    /// Create a new lowerer for the given program
    pub fn new(program: &'a Program) -> Self {
        Self {
            program,
            next_name_id: 0,
            scopes: Vec::new(),
        }
    }

    /// Lower an expression to IR value
    ///
    /// TODO: This function needs significant work:
    /// 1. Move macro expansion here to respect scoping rules
    /// 2. Handle the fact that index operations might actually be parsing generic specializations
    fn lower_value(&mut self, expr: LExpr) -> CResult<TValue> {
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
                // TODO: This is wrong - we might be parsing a generic specialization, not an index
                let (base_expr, args_exprs) = split_postfix(loc.clone(), "index", items)?;
                let base = Box::new(self.lower_value(base_expr)?);
                let mut args = Vec::new();
                for arg in args_exprs {
                    args.push(self.lower_value(arg)?);
                }
                Ok(self.typed_value(loc, Value::Index { base, args }))
            }
            //TODO fix the LHS of this to check for paterns maybe
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

    /// Lower an expression to a pattern for use in match expressions, function params, etc.
    fn lower_pattern(&mut self, expr: LExpr) -> CResult<TPattern> {
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => {
                Ok(self.typed_pattern(expr.loc, Pattern::Wildcard))
            }
            Expr::Atom(Token::Ident(name)) => {
                //TODO decide if this should pass name by value here.
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

    /// Create a typed value with the given location
    fn typed_value(&self, loc: Loc, value: Value) -> TValue {
        Typed {
            loc,
            ty: None,
            value,
        }
    }

    /// Create a typed pattern with the given location
    fn typed_pattern(&self, loc: Loc, value: Pattern) -> TPattern {
        Typed {
            loc,
            ty: None,
            value,
        }
    }

    /// Push a new variable scope onto the stack
    fn push_scope(&mut self) {
        self.scopes.push((HashMap::new(), HashMap::new()));
    }

    /// Pop the current variable scope
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Resolve a name to a NameId, creating a new binding if not found
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

    /// Generate a fresh unique name ID
    fn fresh_name_id(&mut self) -> NameId {
        let id = NameId(self.next_name_id);
        self.next_name_id += 1;
        id
    }

    /// Lower match arms from a block expression or single arm
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

    /// Lower a single match arm (pattern => body or pattern if guard => body)
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

/// Lower a single expression to IR
pub fn lower_expr(program: &Program, expr: LExpr) -> CResult<TValue> {
    IrLowerer::new(program).lower_value(expr)
}

// Helper functions for parsing prefix/postfix constructs

/// Split a prefix expression into its two required arguments
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

/// Split a postfix expression into the target and arguments
fn split_postfix(
    loc: Loc,
    name: &'static str,
    mut items: Vec<LExpr>,
) -> CResult<(LExpr, Vec<LExpr>)> {
    if let Some(first) = items.pop() {
        Ok((first, items))
    } else {
        //TODO this is likely the wrong error message.
        //arity should only aplly to calls
        Err(CompileError::Arity {
            loc,
            call_name: name,
            expected: 1,
            got: 0,
        })
    }
}

