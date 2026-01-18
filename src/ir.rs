use crate::parsing::{Expr, LExpr, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};
use std::collections::HashMap;

// Type aliases for commonly used typed/located constructs
pub type LName = Located<NameId>; // Located name identifier
pub type TName = Typed<NameId>; // Typed name identifier
// pub type TDecl = Typed<Decl>; // Typed declaration
pub type TValue = Typed<Value>; // Typed value/expression
pub type TPattern = Typed<Pattern>; // Typed pattern

// Core type definitions for the IR
#[derive(Debug, Clone, PartialEq)]
pub struct TypeInfo {
    // /// Locations where this type is used in the code
    // pub uses: Vec<Loc>,
}

impl TypeInfo {
    pub fn new_empty()->Self{TypeInfo {}}
}

/// Wrapper that adds location and type information to any value
#[derive(Debug, Clone, PartialEq)]
pub struct Typed<T> {
    /// Source location of this construct
    pub loc: Loc,
    /// Type information (if available/known)
    pub ty: TypeInfo,
    /// The underlying value
    pub value: T,
}

/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub usize);

// /// The intermediate representation (IR) of a complete program
// #[derive(Debug, Clone, PartialEq)]
// pub struct ProgramIr {
//     /// Global/top-level declarations
//     pub decls: Vec<TDecl>,
//     /// Main program body expression
//     pub body: TValue,
// }

/// Generic parameters for declarations
#[derive(Debug, Clone, PartialEq)]
pub struct Generics {
    /// Parameter patterns (typically type variables)
    pub params: Vec<TPattern>,
}

// /// Declaration of a type, constant, or macro in the global scope
// ///
// /// TODO: Eventually support local declarations within functions
// /// TODO: Macros should be handled here (explicitly or conceptually)
// #[derive(Debug, Clone, PartialEq)]
// pub enum Decl {
//     /// Runtime value declaration (constants, functions, etc.)
//     RuntimeValue {
//         name: LName,
//         generics: Option<Generics>,
//         value: TValue,
//     },
//     /// Struct definition with named fields
//     Struct {
//         name: LName,
//         generics: Option<Generics>,
//         fields: Vec<(LName, TPattern)>,
//     },
//     /// Union definition with named fields (one field active at a time)
//     Union {
//         name: LName,
//         generics: Option<Generics>,
//         fields: Vec<(LName, TPattern)>,
//     },
//     /// Enum definition with variants
//     Enum {
//         name: LName,
//         generics: Option<Generics>,
//         variants: Vec<EnumVariant>,
//     },
//     /// Type alias declaration
//     Alias {
//         name: LName,
//         generics: Option<Generics>,
//         ty: TPattern,
//     },
//     /// Macro definition
//     Macro {
//         name: LName,
//         generics: Option<Generics>,
//         params: Vec<LName>,
//         body: TValue,
//     },
// }

/// Single variant within an enum declaration
#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariant {
    pub name: LName,
    pub fields: Vec<TPattern>,
}
/// Pure binary operations.
///
/// Invariant:
/// - No control flow
/// - No short-circuiting
/// - Both operands are evaluated exactly once
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinOp {
    // arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    // bitwise
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // comparisons
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Pure unary operations.
///
/// Invariant:
/// - No control flow
/// - Operand is evaluated exactly once
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnOp {
    Neg,    // -x
    Not,    // !x
    BitNot, // ~x
    Deref,  // *x
    AddrOf, // &x
}

/// Assignment operator, where `None` means plain `=`.
pub type AssignOp = Option<BinOp>;

/// Short-circuiting logical operations.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Or,
}

/// Runtime IR values.
///
/// This IR is *expression-oriented* but *effect-explicit*:
/// - Mutation is represented explicitly (`Assign`, `Let`)
/// - Control flow is explicit (`If`, `While`, `Match`)
/// - `BinOp` / `UnOp` are guaranteed to be pure
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Reference to a resolved name
    NameRef(NameId),

    /// Literal constant
    Literal(Literal),

    /// Function or callable invocation
    Call {
        callee: Box<TValue>,
        args: Vec<TValue>,
    },

    /// Pure binary operation
    BinOp {
        op: BinOp,
        values: Box<(TValue, TValue)>,
    },

    /// Pure unary operation
    UnOp {
        op: UnOp,
        value: Box<TValue>,
    },

    /// Indexing or specialization
    Index {
        base: Box<TValue>,
        args: Vec<TValue>,
    },

    /// Lexical block
    Block {
        statements: Vec<TValue>,
        return_value: Option<Box<TValue>>,
    },

    /// Immutable binding
    Let {
        pat: TPattern,
        value: Box<TValue>,
    },

    /// Assignment with explicit sequencing.
    ///
    /// Not a `BinOp` because:
    /// - LHS is evaluated first
    /// - Mutation occurs
    Assign {
        op: AssignOp,
        target: Box<TValue>,
        value: Box<TValue>,
    },

    /// Short-circuiting logical operations.
    LogicOp {
        op: LogicOp,
        values: Box<(TValue, TValue)>,
    },

    /// Conditional expression
    If {
        cond: Box<TValue>,
        then: Box<TValue>,
        els: Option<Box<TValue>>,
    },

    /// Loop
    While {
        cond: Box<TValue>,
        body: Box<TValue>,
    },

    /// Function literal
    Func {
        params: Vec<Param>,
        ret: Option<TPattern>,
        body: Box<TValue>,
    },

    /// Early return
    Return(Option<Box<TValue>>),

    Break,
    Continue,

    /// Explicit type cast
    Cast {
        value: Box<TValue>,
        ty: TPattern,
    },

    /// Type annotation
    TypeAnnotation {
        value: Box<TValue>,
        ty: TPattern,
    },

    /// Pattern match
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
            ty: TypeInfo::new_empty(),
            value: value.value,
        }
    }
}

// impl ProgramIr {
//     /// Create an empty program IR with a block body
//     pub fn empty(loc: Loc) -> Self {
//         Self {
//             decls: Vec::new(),
//             body: Typed {
//                 loc,
//                 ty: None,
//                 value: Value::Block {
//                     statements: Vec::new(),
//                     return_value: None,
//                 },
//             },
//         }
//     }
// }

impl Program {
    #[inline]
    fn with_scope<T>(&mut self, f: impl FnOnce(&mut Program) -> CResult<T>) -> CResult<T> {
        self.push_scope();
        let result = f(self);
        self.pop_scope();
        result
    }

    //TODO:
    // 1. local macros are intetionaly not handeled and scoping on macros is broken on purpose to be like C
    // 2. some places parse a value where a value/pattern check needs to be done
    pub fn lower_value(&mut self, expr: LExpr) -> CResult<TValue> {
        match expr.value {
            Expr::Atom(token) => self.lower_atom(&expr.loc, token),

            // { ... }
            Expr::Prefix(open, mut items) if open.value == "{" => {
                let loc = expr.loc;

                self.with_scope(|this| {
                    let mut statements = Vec::new();
                    let mut return_value = None;

                    if let Some(last) = items.pop() {
                        for item in items {
                            statements.push(this.lower_value(item)?);
                        }

                        if !matches!(last.value, Expr::Atom(Token::Operator(";"))) {
                            return_value = Some(Box::new(this.lower_value(last)?));
                        }
                    }

                    Ok(this.typed_value(
                        loc,
                        Value::Block {
                            statements,
                            return_value,
                        },
                    ))
                })
            }

            // let <pat> = <value>
            Expr::Prefix(open, mut items) if open.value == "let" => {
                let loc = expr.loc;

                if items.len() < 2 {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: "let expects a pattern and a value",
                    });
                }
                if items.len() > 3 {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: "let has too many parts",
                    });
                }

                let value_expr = items.pop().unwrap();
                let pat_expr = items.pop().unwrap();

                let pat = self.lower_pattern(pat_expr)?;
                let value = Box::new(self.lower_value(value_expr)?);

                Ok(self.typed_value(loc, Value::Let { pat, value }))
            }

            // call: <callee>(args...)
            Expr::Postfix(open, items) if open.value == "(" => {
                let loc = expr.loc;
                debug_assert!(!items.is_empty(), "call expression missing callee");

                let mut it = items.into_iter();
                let callee = Box::new(self.lower_value(it.next().unwrap())?);

                let mut args = Vec::new();
                for arg in it {
                    args.push(self.lower_value(arg)?);
                }

                Ok(self.typed_value(loc, Value::Call { callee, args }))
            }

            // index: <base>[args...]
            Expr::Postfix(open, items) if open.value == "[" => {
                let loc = expr.loc;
                debug_assert!(!items.is_empty(), "index expression missing base");

                let mut it = items.into_iter();
                //TODO this can actually be a generic so a pattern
                let base = Box::new(self.lower_value(it.next().unwrap())?);

                let mut args = Vec::new();
                for arg in it {
                    args.push(self.lower_value(arg)?);
                }

                Ok(self.typed_value(loc, Value::Index { base, args }))
            }

            // match <value> { arms... }
            Expr::Prefix(open, items) if open.value == "match" => {
                let loc = expr.loc;

                if items.len() < 2 {
                    return Err(CompileError::SimpleError {
                        loc,
                        s: "match expects a value and at least one arm",
                    });
                }

                let mut it = items.into_iter();
                let value = Box::new(self.lower_value(it.next().unwrap())?);

                let mut arms = Vec::new();
                for arm in it {
                    arms.push(self.lower_match_arm(arm)?);
                }

                Ok(self.typed_value(loc, Value::Match { value, arms }))
            }

            // assignment
            Expr::Bin(op, pair) if op.value == "=" => {
                let loc = expr.loc;
                let (lhs, rhs) = *pair;

                //TODO: target might be a pattern in rare cases? not sure
                let target = Box::new(self.lower_value(lhs)?);
                let value = Box::new(self.lower_value(rhs)?);

                Ok(self.typed_value(
                    loc,
                    Value::Assign {
                        op: None,
                        target,
                        value,
                    },
                ))
            }

            // fn (sig) body
            Expr::Prefix(open, items) if open.value == "fn" => {
                let loc = expr.loc;

                debug_assert!(
                    (1..=2).contains(&items.len()),
                    "fn expects signature and optional body"
                );

                let mut items = items;

                let body_expr = if items.len() == 2 {
                    items.pop().unwrap()
                } else {
                    todo!()
                };

                let sig_expr = items.pop().expect("fn missing signature");

                let (params_expr, ret_expr) = match sig_expr.value {
                    Expr::Bin(arrow, pair) if arrow.value == "->" => {
                        let (lhs, rhs) = *pair;
                        (lhs, Some(rhs))
                    }
                    _ => (sig_expr, None),
                };

                let Expr::Prefix(p_open, param_items) = params_expr.value else {
                    debug_assert!(false, "fn signature does not start with parameter list");
                    unreachable!();
                };
                debug_assert!(p_open.value == "(", "fn parameter list must start with '('");

                self.with_scope(|p| {
                    let mut params = Vec::with_capacity(param_items.len());
                    for param in param_items {
                        let pat = p.lower_pattern(param)?;
                        params.push(Param { pat, ty: None });
                    }

                    let ret = match ret_expr {
                        Some(e) => Some(p.lower_pattern(e)?),
                        None => None,
                    };

                    let body = Box::new(p.lower_value(body_expr)?);

                    Ok(p.typed_value(loc, Value::Func { params, ret, body }))
                })
            }

            Expr::Prefix(open, items) => self.lower_prefix_op(expr.loc, open, items),

            Expr::Postfix(open, items) => self.lower_postfix_op(expr.loc, open, items),

            Expr::Bin(op, pair) => {
                let (lhs, rhs) = *pair;
                self.lower_binary_op(expr.loc, op.value, lhs, rhs)
            }
        }
    }

    fn lower_atom(&mut self, loc: &Loc, token: Token) -> CResult<TValue> {
        let value = match token {
            Token::NumLit(n) => Value::Literal(Literal::Num(n)),
            Token::FloatLit(f) => Value::Literal(Literal::Float(f)),
            Token::StrLit(s) => Value::Literal(Literal::Str(s)),

            Token::Ident(name) => {
                let id = self.resolve_value(loc, &name)?;
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

    pub fn lower_pattern(&mut self, expr: LExpr) -> CResult<TPattern> {
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => {
                Ok(self.typed_pattern(expr.loc, Pattern::Wildcard))
            }

            Expr::Atom(Token::Ident(name)) => {
                let id = self.insert_value_in_current_scope(&name);
                Ok(self.typed_pattern(expr.loc, Pattern::Bind(id)))
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let mut parts = Vec::with_capacity(items.len());
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

    fn resolve_value(&mut self, loc: &Loc, name: &str) -> CResult<NameId> {
        for (value_scope, _) in self.scopes.iter().rev() {
            if let Some(id) = value_scope.get(name) {
                return Ok(*id);
            }
        }
        Err(CompileError::SimpleError {
            loc: loc.clone(),
            s: "Unresolved name",
        })
    }

    fn lower_match_arm(&mut self, expr: LExpr) -> CResult<MatchArm> {
        match expr.value {
            Expr::Bin(op, pair) if op.value == "=>" => {
                let (pat_expr, body_expr) = *pair;
                let pat = self.lower_pattern(pat_expr)?;
                let body = self.lower_value(body_expr)?;
                Ok(MatchArm {
                    pat,
                    guard: None,
                    body,
                })
            }

            Expr::Bin(op, pair) if op.value == "if" => {
                let (left, guard_expr) = *pair;

                if let Expr::Bin(arrow, inner_pair) = left.value
                    && arrow.value == "=>"
                {
                    let (pat_expr, body_expr) = *inner_pair;
                    let pat = self.lower_pattern(pat_expr)?;
                    let guard = self.lower_value(guard_expr)?;
                    let body = self.lower_value(body_expr)?;
                    Ok(MatchArm {
                        pat,
                        guard: Some(guard),
                        body,
                    })
                } else {
                    Err(CompileError::SimpleError {
                        loc: expr.loc,
                        s: "Invalid match arm guard syntax",
                    })
                }
            }

            _ => Err(CompileError::SimpleError {
                loc: expr.loc,
                s: "Invalid match arm syntax",
            }),
        }
    }

    // ===============================
    // Operator lowering helpers
    // ===============================

    fn lower_prefix_op(
        &mut self,
        loc: Loc,
        op: Located<&str>,
        items: Vec<LExpr>,
    ) -> CResult<TValue> {
        let unop = match op.value {
            "-" => UnOp::Neg,
            "!" => UnOp::Not,
            "~" => UnOp::BitNot,
            "*" => UnOp::Deref,
            "&" => UnOp::AddrOf,

            "++" => return self.lower_inc_dec_prefix(op.map(|_| BinOp::Add), items),
            "--" => return self.lower_inc_dec_prefix(op.map(|_| BinOp::Sub), items),

            _ => {
                return Err(CompileError::SimpleError {
                    loc,
                    s: "Unsupported expression in IR lowering",
                });
            }
        };

        if items.len() != 1 {
            panic!("prefix operator `{}` with {} operands", op, items.len());
        }

        let rhs = Box::new(self.lower_value(items.into_iter().next().unwrap())?);

        Ok(self.typed_value(
            loc,
            Value::UnOp {
                op: unop,
                value: rhs,
            },
        ))
    }

    fn lower_postfix_op(
        &mut self,
        loc: Loc,
        op: Located<&str>,
        items: Vec<LExpr>,
    ) -> CResult<TValue> {
        match op.value {
            // these are handled earlier and must never reach here
            "(" | "[" => unreachable!("call/index should be handled before postfix ops"),

            "++" => self.lower_inc_dec_postfix(op.map(|_| BinOp::Add), items),
            "--" => self.lower_inc_dec_postfix(op.map(|_| BinOp::Sub), items),

            _ => {
                return Err(CompileError::SimpleError {
                    loc,
                    s: "Unsupported expression in IR lowering",
                });
            }
        }
    }

    fn lower_inc_dec_prefix(
        &mut self,
        op: Located<BinOp>,
        mut items: Vec<LExpr>,
    ) -> CResult<TValue> {
        if items.len() != 1 {
            panic!("prefix operator with {} operands", items.len());
        }

        let target = Box::new(self.lower_value(items.pop().unwrap())?);

        Ok(self.typed_value(
            op.loc.clone(),
            Value::Assign {
                op: Some(op.value),
                target,
                value: Box::new(self.typed_value(op.loc, Value::Literal(Literal::Num(1)))),
            },
        ))
    }

    fn lower_inc_dec_postfix(
        &mut self,
        op: Located<BinOp>,
        mut items: Vec<LExpr>,
    ) -> CResult<TValue> {
        if items.len() != 1 {
            panic!("postfix operator with {} operands", items.len());
        }

        let target = Box::new(self.lower_value(items.remove(0))?);

        Ok(self.typed_value(
            op.loc.clone(),
            Value::Assign {
                op: Some(op.value),
                target,
                value: Box::new(self.typed_value(op.loc, Value::Literal(Literal::Num(1)))),
            },
        ))
    }

    fn lower_binary_op(&mut self, loc: Loc, op: &str, lhs: LExpr, rhs: LExpr) -> CResult<TValue> {
        if let Some(assign_op) = match op {
            "=" => Some(None),
            "+=" => Some(Some(BinOp::Add)),
            "-=" => Some(Some(BinOp::Sub)),
            "*=" => Some(Some(BinOp::Mul)),
            "/=" => Some(Some(BinOp::Div)),
            "%=" => Some(Some(BinOp::Mod)),
            "&=" => Some(Some(BinOp::BitAnd)),
            "|=" => Some(Some(BinOp::BitOr)),
            "^=" => Some(Some(BinOp::BitXor)),
            "<<=" => Some(Some(BinOp::Shl)),
            ">>=" => Some(Some(BinOp::Shr)),
            _ => None,
        } {
            let target = Box::new(self.lower_value(lhs)?);
            let value = Box::new(self.lower_value(rhs)?);
            return Ok(self.typed_value(
                loc,
                Value::Assign {
                    op: assign_op,
                    target,
                    value,
                },
            ));
        }

        if let Some(logic_op) = match op {
            "&&" => Some(LogicOp::And),
            "||" => Some(LogicOp::Or),
            _ => None,
        } {
            let left = self.lower_value(lhs)?;
            let right = self.lower_value(rhs)?;
            return Ok(self.typed_value(
                loc,
                Value::LogicOp {
                    op: logic_op,
                    values: Box::new((left, right)),
                },
            ));
        }

        let binop = match op {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            "%" => BinOp::Mod,

            "&" => BinOp::BitAnd,
            "|" => BinOp::BitOr,
            "^" => BinOp::BitXor,
            "<<" => BinOp::Shl,
            ">>" => BinOp::Shr,

            "==" => BinOp::Eq,
            "!=" => BinOp::Ne,
            "<" => BinOp::Lt,
            "<=" => BinOp::Le,
            ">" => BinOp::Gt,
            ">=" => BinOp::Ge,

            // "~" | "!" =>
            //     panic!("operator `{}` cannot appear as binary op (parser bug)", op),
            _ => {
                return Err(CompileError::SimpleError {
                    loc,
                    s: "Unsupported expression in IR lowering",
                });
            }
        };

        let left = self.lower_value(lhs)?;
        let right = self.lower_value(rhs)?;

        Ok(self.typed_value(
            loc,
            Value::BinOp {
                op: binop,
                values: Box::new((left, right)),
            },
        ))
    }

    /// Create a typed value with the given location
    fn typed_value(&self, loc: Loc, value: Value) -> TValue {
        Typed {
            loc,
            ty: TypeInfo::new_empty(),
            value,
        }
    }

    /// Create a typed pattern with the given location
    fn typed_pattern(&self, loc: Loc, value: Pattern) -> TPattern {
        Typed {
            loc,
            ty: TypeInfo::new_empty(),
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

    /// Insert a new binding into the current (innermost) scope, always creating a fresh ID.
    fn insert_value_in_current_scope(&mut self, name: &str) -> NameId {
        let id = self.fresh_name_id();
        if let Some((value_scope, _)) = self.scopes.last_mut() {
            value_scope.insert(name.to_string(), id);
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
}

// Public API functions

#[cfg(test)]
mod var_scope_test {
    use super::*;
    use crate::parsing::Parser;
    use crate::program::Program;
    #[test]
    fn var_scope_is_respected() {
        // { let a = 1; { let a = 2; a } a }
        let src = "{ let a = 1; { let a = 2; a; } a; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().expect("failed to parse expr");
        let ir = program.lower_value(expr).expect("lowering failed");
        // top-level should be a block
        let top_block = match ir.value {
            Value::Block { ref statements, .. } => statements,
            _ => panic!("expected top-level block"),
        };
        // expect three statements: let a, inner block, final a
        assert_eq!(top_block.len(), 3);
        // Grab outer let bind id
        let outer_pat = match &top_block[0].value {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected outer let"),
        };
        let outer_id = match &outer_pat.value {
            Pattern::Bind(id) => *id,
            _ => panic!("expected bind pattern"),
        };
        // Final name ref refers to outer
        let final_ref = match &top_block[2].value {
            Value::NameRef(id) => *id,
            _ => panic!("expected final name reference"),
        };
        assert_eq!(outer_id, final_ref);
        // Inner block
        let inner_block = match &top_block[1].value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected inner block"),
        };
        assert_eq!(inner_block.len(), 2);
        let inner_pat = match &inner_block[0].value {
            Value::Let { pat, .. } => pat,
            _ => panic!("expected inner let"),
        };
        let inner_id = match &inner_pat.value {
            Pattern::Bind(id) => *id,
            _ => panic!("expected bind pattern"),
        };
        let inner_ref = match &inner_block[1].value {
            Value::NameRef(id) => *id,
            _ => panic!("expected inner name reference"),
        };
        assert_eq!(inner_id, inner_ref);
        // Ensure inner and outer ids differ
        assert_ne!(
            outer_id, inner_id,
            "inner and outer bindings must not collide"
        );
    }
}

#[cfg(test)]
mod lowering_tests {
    use super::*;
    use crate::parsing::Parser;
    use crate::program::{CompileError, Program};

    fn lower_block(src: &str) -> TValue {
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        program.lower_value(expr).unwrap()
    }

    fn bound_id(stmt: &TValue) -> NameId {
        match &stmt.value {
            Value::Let { pat, .. } => match pat.value {
                Pattern::Bind(id) => id,
                _ => panic!("expected bind pattern"),
            },
            _ => panic!("expected let statement"),
        }
    }

    #[test]
    fn lowers_call_and_index_with_bound_names() {
        let src = "{ let f = 1; let a = 2; f(a, 3)[a, 4]; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 3);

        let f_id = bound_id(&statements[0]);
        let a_id = bound_id(&statements[1]);

        let (base, index_args) = match &statements[2].value {
            Value::Index { base, args } => (base, args),
            _ => panic!("expected index expression"),
        };

        let (callee, call_args) = match &base.value {
            Value::Call { callee, args } => (callee, args),
            _ => panic!("expected call base"),
        };

        match callee.value {
            Value::NameRef(id) => assert_eq!(id, f_id),
            _ => panic!("expected callee to be name"),
        }
        assert_eq!(call_args.len(), 2);
        match call_args[0].value {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first call arg to be name"),
        }
        match call_args[1].value {
            Value::Literal(Literal::Num(3)) => {}
            _ => panic!("expected literal call arg"),
        }

        assert_eq!(index_args.len(), 2);
        match index_args[0].value {
            Value::NameRef(id) => assert_eq!(id, a_id),
            _ => panic!("expected first index arg to be name"),
        }
        match index_args[1].value {
            Value::Literal(Literal::Num(4)) => {}
            _ => panic!("expected literal index arg"),
        }
    }

    #[test]
    fn lowers_match_with_wildcard_arm() {
        let src = "{ let x = 1; match x { _ => x; }; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        let x_id = bound_id(&statements[0]);

        let (scrutinee, arms) = match &statements[1].value {
            Value::Match { value, arms } => (value, arms),
            _ => panic!("expected match"),
        };

        match scrutinee.value {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected scrutinee name"),
        }

        assert_eq!(arms.len(), 1);
        match arms[0].pat.value {
            Pattern::Wildcard => {}
            _ => panic!("expected wildcard pattern"),
        }
        match arms[0].body.value {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected arm body to reference x"),
        }
    }

    #[test]
    fn lowers_match_as_block_return_value() {
        let src = "{ let x = 1; match x { _ => x; } }";
        let ir = lower_block(src);

        let (statements, return_value) = match ir.value {
            Value::Block {
                statements,
                return_value,
            } => (statements, return_value),
            _ => panic!("expected block"),
        };

        assert_eq!(statements.len(), 1);
        assert!(return_value.is_some());

        let x_id = bound_id(&statements[0]);
        let match_expr = return_value.unwrap();
        let (scrutinee, arms) = match match_expr.value {
            Value::Match { value, arms } => (value, arms),
            _ => panic!("expected match"),
        };

        match scrutinee.value {
            Value::NameRef(id) => assert_eq!(id, x_id),
            _ => panic!("expected scrutinee name"),
        }
        assert_eq!(arms.len(), 1);
    }

    #[test]
    fn lowers_compound_assign_inc_and_logic_ops() {
        let src = "{ let a = 1; a += 2; ++a; a++; a && a; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 5);

        let a_id = bound_id(&statements[0]);

        match &statements[1].value {
            Value::Assign { op, target, value } => {
                assert_eq!(*op, Some(BinOp::Add));
                match target.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected assign target name"),
                }
                match value.value {
                    Value::Literal(Literal::Num(2)) => {}
                    _ => panic!("expected assign literal value"),
                }
            }
            _ => panic!("expected compound assignment"),
        }

        for stmt in [&statements[2], &statements[3]] {
            match &stmt.value {
                Value::Assign { op, target, value } => {
                    assert_eq!(*op, Some(BinOp::Add));
                    match target.value {
                        Value::NameRef(id) => assert_eq!(id, a_id),
                        _ => panic!("expected inc target name"),
                    }
                    match value.value {
                        Value::Literal(Literal::Num(1)) => {}
                        _ => panic!("expected inc literal value"),
                    }
                }
                _ => panic!("expected inc assignment"),
            }
        }

        match &statements[4].value {
            Value::LogicOp { op, values } => {
                assert_eq!(*op, LogicOp::And);
                let (left, right) = &**values;
                match left.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected logic left name"),
                }
                match right.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected logic right name"),
                }
            }
            _ => panic!("expected logic op"),
        }
    }

    #[test]
    fn unsupported_expression_reports_error() {
        let src = "if 1 2";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        let err = program.lower_value(expr).unwrap_err();
        match err {
            CompileError::SimpleError { s, .. } => {
                assert_eq!(s, "Unsupported expression in IR lowering");
            }
            _ => panic!("expected simple error"),
        }
    }
}
