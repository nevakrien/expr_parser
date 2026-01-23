use crate::error_messages::{
    ERR_ACCESS_EXPECTS_NAME, ERR_INVALID_MATCH_ARM, ERR_INVALID_MATCH_ARM_GUARD,
    ERR_MATCH_ARM_NEEDS_VALUE, ERR_UNRESOLVED_NAME, ERR_UNSUPPORTED_EXPRESSION,
    ERR_UNSUPPORTED_EXPRESSION_ATOM, ERR_UNSUPPORTED_PATTERN,
};
use crate::parsing::{Expr, LExpr, Loc, Located, Token};
use crate::program::{CResult, CompileError, Program};

// Type aliases for commonly used typed/located constructs
pub type LName = Located<NameId>; // Located name identifier
pub type TName = Typed<NameId>; // Typed name identifier
// pub type TDecl = Typed<Decl>; // Typed declaration
pub type TValue = Typed<Value>; // Typed value/expression
pub type LValue = Located<Value>; // Typed value/expression
pub type TPattern = Typed<Pattern>; // Typed pattern

// Core type definitions for the IR
#[derive(Debug, Clone, PartialEq)]
pub struct TypeInfo {
    // /// Locations where this type is used in the code
    pub uses: Vec<Located<TypeUse>>,
}

impl TypeInfo {
    pub fn new_empty() -> Self {
        Self { uses: Vec::new() }
    }
}

// #[derive(Debug,Copy, Clone, PartialEq)]
// pub struct TypeId(usize);

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum InferId {
    Concrete(usize),
    Infered(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeUse {
    FuncOutputs(InferId),
    FuncInputs(Box<[InferId]>),
    Tuple(Box<[InferId]>),
    Basic(NameId),
}

/// Wrapper that adds location and type information to any value
#[derive(Debug, Clone, PartialEq)]
pub struct Typed<T> {
    /// Source location of this construct
    pub loc: Loc,
    /// Type information (if available/known)
    pub ty: InferId,
    /// The underlying value
    pub value: T,
}

/// Unique identifier for names in the IR
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct NameId(pub usize);

/// Literal values that can appear in the code
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Num(u64),
    Float(f64),
    Str(String),
    Void,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessKind {
    Dot,
    Type,
    Ptr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessName {
    pub name: String,
}

/// Function parameter declaration
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub pat: TPattern,
    pub ty: Option<TValue>,
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

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum Dir {
    Inc,
    Dec,
}

/// Assignment operator, where `None` means plain `=`.
#[derive(Debug, Clone, PartialEq)]
pub enum AssignOp {
    Nothing(Box<TValue>),
    Bin(BinOp, Box<TValue>),
    Pre(Dir),
    Post(Dir),
}

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

    /// Wildcard pattern that matches anything (_)
    Wildcard,

    Tuple(Vec<TValue>),

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

    //===== TYPES =====
    /// Explicit type cast
    Cast {
        value: Box<TValue>,
        ty: Box<TValue>,
    },

    /// Type annotation
    TypeAnnotation {
        value: Box<TValue>,
        ty: Box<TValue>,
    },

    //==== MUTATION GATES =====
    /// Function or callable invocation
    Call {
        callee: Box<TValue>,
        args: Vec<TValue>,
    },

    /// Assignment with explicit sequencing.
    ///
    /// Not a `BinOp` because:
    /// - LHS is evaluated first
    /// - Mutation occurs
    Assign {
        op: AssignOp,
        target: Box<TValue>,
    },

    /// Indexing or specialization
    Index {
        base: Box<TValue>,
        args: Vec<TValue>,
    },

    /// Field/type access with deferred name resolution
    Access {
        base: Box<TValue>,
        name: AccessName,
        kind: AccessKind,
    },

    // ===== SCOPE =====
    /// Immutable binding
    Let {
        pat: TPattern,
        value: Box<TValue>,
        else_part: Option<Box<TValue>>,
    },

    /// Lexical block
    Block {
        statements: Vec<TValue>,
        return_value: Option<Box<TValue>>,
    },

    //==== CONTROL FLOW =====
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

    /// Pattern match
    Match {
        value: Box<TValue>,
        arms: Vec<MatchArm>,
    },
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
    /// Literal value pattern
    Literal(Literal),
    /// Type annotation pattern (x:T)
    TypeAnnotation { pat: Box<TPattern>, ty: Box<TValue> },
    //==== TODOS: ========

    // /// Struct/enum destructoring pattern
    // Destructure {
    //     ctor: LName,
    //     fields: Vec<PatternField>,
    // },
    // /// Generic type specialization (e.g., Foo[T, U])
    // GenericSpecialization {
    //     base: Box<TPattern>,
    //     args: Vec<TPattern>,
    // },
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
            ty: self.ty,
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

// impl<T> From<Located<T>> for Typed<T> {
//     fn from(value: Located<T>) -> Self {
//         Typed {
//             loc: value.loc,
//             ty: TypeInfo::new_empty(),
//             value: value.value,
//         }
//     }
// }

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
            Expr::Prefix(open, items) if open.value == "{" => {
                self.lower_block_expr(expr.loc, items)
            }

            // let <pat> = <value>
            Expr::Prefix(open, items) if open.value == "let" => {
                self.lower_let_expr(expr.loc, items)
            }

            // call: <callee>(args...)
            Expr::Postfix(open, items) if open.value == "(" => {
                self.lower_call_expr(expr.loc, items)
            }

            // index: <base>[args...]
            Expr::Postfix(open, items) if open.value == "[" => {
                self.lower_index_expr(expr.loc, items)
            }

            // match <value> { arms... }
            Expr::Prefix(open, items) if open.value == "match" => {
                self.lower_match_expr(expr.loc, items)
            }

            // assignment
            Expr::Bin(op, pair) if op.value == "=" => self.lower_assign_expr(expr.loc, *pair),

            // fn (sig) body
            Expr::Prefix(open, items) if open.value == "fn" => self.lower_fn_expr(expr.loc, items),

            Expr::Bin(op, pair) if (op.value == "as" || op.value == ":") => {
                self.lower_cast_expr(expr.loc, op, *pair)
            }

            Expr::Bin(op, pair) if matches!(op.value, "." | "::" | "->") => {
                let (lhs, rhs) = *pair;
                self.lower_access_expr(expr.loc, op, lhs, rhs)
            }

            //fallbacks
            Expr::Prefix(open, items) => self.lower_prefix_op(expr.loc, open, items),
            Expr::Postfix(open, items) => self.lower_postfix_op(expr.loc, open, items),
            Expr::Bin(op, pair) => {
                let (lhs, rhs) = *pair;
                self.lower_binary_op(expr.loc, op, lhs, rhs)
            }
        }
    }

    #[inline(always)]
    fn lower_atom(&mut self, loc: &Loc, token: Token) -> CResult<TValue> {
        let value = match token {
            Token::NumLit(n) => Value::Literal(Literal::Num(n)),
            Token::FloatLit(f) => Value::Literal(Literal::Float(f)),
            Token::StrLit(s) => Value::Literal(Literal::Str(s)),
            Token::Operator("(") => Value::Literal(Literal::Void),

            Token::Ident(name) => {
                let id = self.resolve_value(loc, &name)?;
                Value::NameRef(id)
            }

            Token::Operator(op) => {
                return Err(CompileError::UnsupportedForm {
                    loc: loc.clone(),
                    op_loc: Some(loc.clone()),
                    op: Some(op),
                    message: ERR_UNSUPPORTED_EXPRESSION_ATOM,
                });
            }
        };

        Ok(self.typed_value(loc.clone(), value))
    }

    #[inline(always)]
    fn lower_block_expr(&mut self, loc: Loc, mut items: Vec<LExpr>) -> CResult<TValue> {
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

    #[inline(always)]
    fn lower_let_expr(&mut self, loc: Loc, mut items: Vec<LExpr>) -> CResult<TValue> {
        debug_assert!(2 <= items.len() && items.len() <= 3);

        let else_exp = if items.len() == 3 { items.pop() } else { None };

        let value_expr = items.pop().unwrap();
        let pat_expr = items.pop().unwrap();

        let value = Box::new(self.lower_value(value_expr)?);
        let pat = self.lower_pattern(pat_expr)?;

        let else_part = if let Some(exp) = else_exp {
            let v = self.with_scope(|prog| prog.lower_value(exp))?;
            Some(Box::new(v))
        } else {
            None
        };

        Ok(self.typed_value(
            loc,
            Value::Let {
                pat,
                value,
                else_part,
            },
        ))
    }

    #[inline(always)]
    fn lower_call_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<TValue> {
        debug_assert!(!items.is_empty(), "call expression missing callee");

        let mut items = items;
        let callee = Box::new(self.lower_value(items.remove(0))?);

        let args: Result<Vec<_>, _> = items.into_iter().map(|arg| self.lower_value(arg)).collect();
        let args = args?;

        Ok(self.typed_value(loc, Value::Call { callee, args }))
    }

    #[inline(always)]
    fn lower_index_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<TValue> {
        debug_assert!(!items.is_empty(), "index expression missing base");

        let mut items = items;
        //TODO this can actually be a generic so a pattern
        let base = Box::new(self.lower_value(items.remove(0))?);

        let args: Result<Vec<_>, _> = items.into_iter().map(|arg| self.lower_value(arg)).collect();
        let args = args?;

        Ok(self.typed_value(loc, Value::Index { base, args }))
    }

    #[inline(always)]
    fn lower_match_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<TValue> {
        if items.len() < 2 {
            return Err(CompileError::SimpleError {
                loc,
                s: ERR_MATCH_ARM_NEEDS_VALUE,
            });
        }

        let mut items = items;
        let value = Box::new(self.lower_value(items.remove(0))?);

        let arms: Result<Vec<_>, _> = items
            .into_iter()
            .map(|arm| self.lower_match_arm(arm))
            .collect();
        let arms = arms?;

        Ok(self.typed_value(loc, Value::Match { value, arms }))
    }

    #[inline(always)]
    fn lower_assign_expr(&mut self, loc: Loc, pair: (LExpr, LExpr)) -> CResult<TValue> {
        let (lhs, rhs) = pair;

        //TODO: target might be a pattern in rare cases? not sure
        let target = Box::new(self.lower_value(lhs)?);
        let value = Box::new(self.lower_value(rhs)?);

        Ok(self.typed_value(
            loc,
            Value::Assign {
                op: AssignOp::Nothing(value),
                target,
            },
        ))
    }

    #[inline(always)]
    fn lower_fn_expr(&mut self, loc: Loc, items: Vec<LExpr>) -> CResult<TValue> {
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
                //TODO support type anotation
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

    #[inline(always)]
    fn lower_cast_expr(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        pair: (LExpr, LExpr),
    ) -> CResult<TValue> {
        let (value_expr, ty_expr) = pair;
        let value = Box::new(self.lower_value(value_expr)?);
        let ty = Box::new(self.lower_value(ty_expr)?);
        let v = match op.value {
            "as" => Value::Cast { value, ty },
            ":" => Value::TypeAnnotation { value, ty },
            _ => panic!("unsupported cast operator `{}`", op.value),
        };
        Ok(self.typed_value(loc, v))
    }

    #[inline(always)]
    fn lower_access_expr(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        lhs: LExpr,
        rhs: LExpr,
    ) -> CResult<TValue> {
        let base = Box::new(self.lower_value(lhs)?);
        let name = match rhs.value {
            Expr::Atom(Token::Ident(name)) => AccessName { name },
            _ => {
                return Err(CompileError::SimpleError {
                    loc: rhs.loc,
                    s: ERR_ACCESS_EXPECTS_NAME,
                });
            }
        };

        let kind = match op.value {
            "." => AccessKind::Dot,
            "::" => AccessKind::Type,
            "->" => AccessKind::Ptr,
            _ => panic!("unsupported access operator `{}`", op.value),
        };

        Ok(self.typed_value(loc, Value::Access { base, name, kind }))
    }

    pub fn lower_pattern(&mut self, expr: LExpr) -> CResult<TPattern> {
        let loc = expr.loc;
        match expr.value {
            Expr::Atom(Token::Ident(name)) if name == "_" => {
                Ok(self.typed_pattern(loc, Pattern::Wildcard))
            }

            Expr::Atom(Token::Ident(name)) => {
                let id = self.insert_value_in_current_scope(name);
                Ok(self.typed_pattern(loc, Pattern::Bind(id)))
            }

            Expr::Prefix(open, items) if open.value == "(" => {
                let mut parts = Vec::with_capacity(items.len());
                for item in items {
                    parts.push(self.lower_pattern(item)?);
                }
                Ok(self.typed_pattern(loc, Pattern::Tuple(parts)))
            }

            // Pattern with type annotation: x:T
            Expr::Bin(op, pair) if op.value == ":" => {
                let (pat_expr, ty_expr) = *pair;
                let pat = self.lower_pattern(pat_expr)?;
                let ty = self.lower_value(ty_expr)?;

                // Create a type annotation pattern
                Ok(self.typed_pattern(
                    loc,
                    Pattern::TypeAnnotation {
                        pat: Box::new(pat),
                        ty: Box::new(ty),
                    },
                ))
            }

            Expr::Bin(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_PATTERN,
            }),

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_PATTERN,
            }),

            _ => Err(CompileError::UnsupportedForm {
                loc,
                op_loc: None,
                op: None,
                message: ERR_UNSUPPORTED_PATTERN,
            }),
        }
    }

    fn resolve_value(&mut self, loc: &Loc, name: &str) -> CResult<NameId> {
        for value_scope in self.scopes.iter().rev() {
            if let Some(id) = value_scope.get(name) {
                return Ok(*id);
            }
        }
        Err(CompileError::SimpleError {
            loc: loc.clone(),
            s: ERR_UNRESOLVED_NAME,
        })
    }

    #[inline(always)]
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
                    Err(CompileError::UnsupportedForm {
                        loc: expr.loc,
                        op_loc: Some(op.loc),
                        op: Some(op.value),
                        message: ERR_INVALID_MATCH_ARM_GUARD,
                    })
                }
            }

            Expr::Bin(op, _) => Err(CompileError::UnsupportedForm {
                loc: expr.loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_INVALID_MATCH_ARM,
            }),

            Expr::Prefix(op, _) | Expr::Postfix(op, _) => Err(CompileError::UnsupportedForm {
                loc: expr.loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_INVALID_MATCH_ARM,
            }),

            _ => Err(CompileError::UnsupportedForm {
                loc: expr.loc,
                op_loc: None,
                op: None,
                message: ERR_INVALID_MATCH_ARM,
            }),
        }
    }

    // ===============================
    // Operator lowering helpers
    // ===============================

    #[inline(always)]
    fn lower_prefix_op(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        items: Vec<LExpr>,
    ) -> CResult<TValue> {
        let unop = match op.value {
            "-" => UnOp::Neg,
            "!" => UnOp::Not,
            "~" => UnOp::BitNot,
            "*" => UnOp::Deref,
            "&" => UnOp::AddrOf,

            "++" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Inc), items),
            "--" => return self.lower_inc_dec_prefix(op.map(|_| Dir::Dec), items),

            _ => {
                return Err(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: ERR_UNSUPPORTED_EXPRESSION,
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

    #[inline(always)]
    fn lower_postfix_op(
        &mut self,
        _loc: Loc,
        op: Located<&'static str>,
        items: Vec<LExpr>,
    ) -> CResult<TValue> {
        match op.value {
            // these are handled earlier and must never reach here
            "(" | "[" => unreachable!("call/index should be handled before postfix ops"),

            "++" => self.lower_inc_dec_postfix(op.map(|_| Dir::Inc), items),
            "--" => self.lower_inc_dec_postfix(op.map(|_| Dir::Dec), items),

            _ => Err(CompileError::UnsupportedForm {
                loc: _loc,
                op_loc: Some(op.loc),
                op: Some(op.value),
                message: ERR_UNSUPPORTED_EXPRESSION,
            }),
        }
    }

    #[inline(always)]
    fn lower_inc_dec_prefix(&mut self, op: Located<Dir>, mut items: Vec<LExpr>) -> CResult<TValue> {
        if items.len() != 1 {
            panic!("prefix operator with {} operands", items.len());
        }

        let target = Box::new(self.lower_value(items.pop().unwrap())?);

        Ok(self.typed_value(
            op.loc,
            Value::Assign {
                op: AssignOp::Pre(op.value),
                target,
            },
        ))
    }

    #[inline(always)]
    fn lower_inc_dec_postfix(
        &mut self,
        op: Located<Dir>,
        mut items: Vec<LExpr>,
    ) -> CResult<TValue> {
        if items.len() != 1 {
            panic!("postfix operator with {} operands", items.len());
        }

        let target = Box::new(self.lower_value(items.remove(0))?);

        Ok(self.typed_value(
            op.loc.clone(),
            Value::Assign {
                op: AssignOp::Post(op.value),
                target,
            },
        ))
    }

    #[inline(always)]
    fn lower_binary_op(
        &mut self,
        loc: Loc,
        op: Located<&'static str>,
        lhs: LExpr,
        rhs: LExpr,
    ) -> CResult<TValue> {
        if let Some(assign_op) = match op.value {
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
                    target,
                    op: if let Some(o) = assign_op {
                        AssignOp::Bin(o, value)
                    } else {
                        AssignOp::Nothing(value)
                    },
                },
            ));
        }

        if let Some(logic_op) = match op.value {
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

        let binop = match op.value {
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

            _ => {
                return Err(CompileError::UnsupportedForm {
                    loc,
                    op_loc: Some(op.loc),
                    op: Some(op.value),
                    message: ERR_UNSUPPORTED_EXPRESSION,
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
    fn typed_value(&mut self, loc: Loc, value: Value) -> TValue {
        Typed {
            loc,
            ty: self.new_infer_id(),
            value,
        }
    }

    /// Create a typed pattern with the given location
    fn typed_pattern(&mut self, loc: Loc, value: Pattern) -> TPattern {
        Typed {
            loc,
            ty: self.new_infer_id(),
            value,
        }
    }

    fn new_infer_id(&mut self) -> InferId {
        let id = self.current_infrence.len();
        self.current_infrence.push(TypeInfo::new_empty());
        InferId::Infered(id)
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
    use crate::error_messages::{ERR_ACCESS_EXPECTS_NAME, ERR_UNSUPPORTED_EXPRESSION};
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
    fn lowers_access_for_dot_and_paths() {
        let src = "{ let a = 1; let t = 2; a.b; t::c; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 4);

        let a_id = bound_id(&statements[0]);
        let t_id = bound_id(&statements[1]);

        match &statements[2].value {
            Value::Access { base, name, kind } => {
                assert_eq!(*kind, AccessKind::Dot);
                match base.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected dot base name"),
                }
                assert_eq!(name.name, "b");
            }
            _ => panic!("expected dot access"),
        }

        match &statements[3].value {
            Value::Access { base, name, kind } => {
                assert_eq!(*kind, AccessKind::Type);
                match base.value {
                    Value::NameRef(id) => assert_eq!(id, t_id),
                    _ => panic!("expected type base name"),
                }
                assert_eq!(name.name, "c");
            }
            _ => panic!("expected type access"),
        }
    }

    #[test]
    fn access_requires_identifier_rhs() {
        let src = "{ let a = 1; a->++a; }";
        let mut parser = Parser::new(src, 0);
        let mut program = Program::new();
        let expr = parser.consume_expr().unwrap();
        let err = program.lower_value(expr).unwrap_err();
        match err {
            CompileError::SimpleError { s, .. } => {
                assert_eq!(s, ERR_ACCESS_EXPECTS_NAME);
            }
            _ => panic!("expected simple error"),
        }
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
            Value::Assign { op, target } => {
                match op {
                    AssignOp::Bin(bin_op, value) => {
                        assert_eq!(*bin_op, BinOp::Add);
                        match value.value {
                            Value::Literal(Literal::Num(2)) => {}
                            _ => panic!("expected assign literal value"),
                        }
                    }
                    _ => panic!("expected compound assignment op"),
                }
                match target.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected assign target name"),
                }
            }
            _ => panic!("expected compound assignment"),
        }

        match &statements[2].value {
            Value::Assign { op, target } => {
                assert!(matches!(op, AssignOp::Pre(Dir::Inc)));
                match target.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected inc target name"),
                }
            }
            _ => panic!("expected prefix inc assignment"),
        }

        match &statements[3].value {
            Value::Assign { op, target } => {
                assert!(matches!(op, AssignOp::Post(Dir::Inc)));
                match target.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected inc target name"),
                }
            }
            _ => panic!("expected postfix inc assignment"),
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
    fn lowers_cast_expression() {
        let src = "{ let a = 1; a as int; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 2);

        let a_id = bound_id(&statements[0]);
        let cast_expr = &statements[1];
        match &cast_expr.value {
            Value::Cast { value, ty } => {
                match value.value {
                    Value::NameRef(id) => assert_eq!(id, a_id),
                    _ => panic!("expected cast value to be name"),
                }
                match ty.value {
                    Value::NameRef(id) => assert_ne!(id, a_id),
                    _ => panic!("expected cast type pattern"),
                }
            }
            _ => panic!("expected cast expression"),
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
            CompileError::UnsupportedForm { message, .. } => {
                assert_eq!(message, ERR_UNSUPPORTED_EXPRESSION);
            }
            _ => panic!("expected simple error"),
        }
    }

    #[test]
    fn lowers_pattern_with_type_annotation() {
        let src = "{ let x:int = 1; }";
        let ir = lower_block(src);

        let statements = match ir.value {
            Value::Block { statements, .. } => statements,
            _ => panic!("expected block"),
        };
        assert_eq!(statements.len(), 1);

        let let_stmt = &statements[0];
        match &let_stmt.value {
            Value::Let {
                pat,
                value,
                else_part,
            } => {
                assert!(else_part.is_none(), "expected no else part");
                match &pat.value {
                    Pattern::TypeAnnotation { pat: inner_pat, ty } => {
                        // The inner pattern should bind a new name 'x'
                        match &inner_pat.value {
                            Pattern::Bind(_x_id) => {
                                // Verify the value is the expected literal
                                match &value.value {
                                    Value::Literal(Literal::Num(1)) => {}
                                    _ => panic!("expected literal value"),
                                }
                            }
                            _ => panic!("expected bind pattern for variable name"),
                        }
                        // The type should resolve to the predefined 'int' name
                        match &ty.value {
                            Value::NameRef(_int_id) => {} // Type should be a name reference to 'int'
                            _ => panic!("expected type to be name reference to predefined type"),
                        }
                    }
                    _ => panic!("expected type annotation pattern"),
                }
            }
            _ => panic!("expected let statement"),
        }
    }
}
